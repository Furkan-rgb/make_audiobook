"""A four-tab local frontend over the existing pipeline.

The tabs mirror how a narrator voice comes to exist and gets used: design one
from a description, clone one from a recording, narrate a book with it, and
play what came out.  They stay independent because what connects them is files
on disk rather than shared session state: a voice is a directory under
``voices/``, a prepared book is a JSON artifact under ``output/``.  Anything
made here is usable from the CLI, and anything the CLI made shows up here.

Every long handler is a generator that yields a growing log, so a run that
takes an hour reports progress instead of appearing to hang.  GPU work is
serialised behind one lock; see :mod:`.runtime`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from ..config import (
    ACTIVE_VOICE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_BOOK_PATH,
    DEFAULT_PREPARATION_MODEL,
    DEFAULT_PREPARATION_PROVIDER,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    DEFAULT_SYNTHESIS_PROVIDER,
    LANGUAGE,
    NARRATION_INSTRUCTION,
    LOCAL_TTS_MODEL_PATH,
    REFERENCE_TRANSCRIBE,
    TTS_MODEL,
    VOICE_DESIGN_INSTRUCT,
    VOICE_REFERENCE_AUDIO_FILENAME,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICE_REFERENCE_TEXT,
    VOICES_DIR,
)
from ..extraction import SUPPORTED_SOURCE_SUFFIXES
from ..preparation import provider_descriptor, provider_descriptors
from ..synthesis.providers import synthesis_descriptor
from ..synthesis.voices import describe, resolve_voice
from ..workflow import (
    NarrationWorkflowOptions,
    PreparationWorkflowOptions,
    narrate_prepared_script,
    prepare_narration_script,
    prepared_markdown_path,
    resolve_script_path,
    write_prepared_markdown,
)
from .library import (
    delete_voice,
    find_voice,
    import_recording,
    list_audiobooks,
    list_prepared_scripts,
    list_voices,
    rename_voice,
    save_transcript,
)
from .review import flagged_units, load_artifact, render_unit, summarize
from .runtime import (
    gpu_slot,
    loaded_model_name,
    stream_output,
    synthesis_provider,
    unload_model,
)

AUDITION_TEXT = (
    "By morning, the decision no longer seemed complicated. The house was "
    "silent, and pale light crossed the hallway floor."
)

# Every action that occupies the GPU or a long-lived model, with the label it
# shows while it is running.  One list because there is one machine: a
# narration and a voice audition cannot both have it, and preparation holds a
# local LLM in the same VRAM the TTS checkpoint needs.  Disabling all of them
# together turns a silent wait on the lock into a visible one.
HEAVY_ACTIONS = (
    ("Free GPU", "Freeing..."),
    ("Generate", "Generating..."),
    ("Process recording", "Processing..."),
    ("Audition", "Auditioning..."),  # clone tab
    ("Infer transcript again", "Transcribing..."),
    ("Audition", "Auditioning..."),  # library tab
    ("Prepare", "Preparing..."),
    ("Narrate", "Narrating..."),
)


def _begin_work(index: int, *clears):
    """Take the machine: disable every heavy action and blank stale results.

    The clears matter as much as the disabling.  An audio player still holding
    the previous take, or a Review header still describing the previous
    artifact, is worse than an empty one — it invites a judgement about work
    that has not happened yet.
    """

    def handler():
        updates = [
            gr.update(interactive=False, value=busy)
            if position == index
            else gr.update(interactive=False)
            for position, (_, busy) in enumerate(HEAVY_ACTIONS)
        ]
        return updates + list(clears)

    return handler


def _end_work():
    """Give it back.  Wired with .then, so a crashed run still restores the UI."""

    return [gr.update(interactive=True, value=idle) for idle, _ in HEAVY_ACTIONS]


def _voice_choices() -> list[tuple[str, str]]:
    return [(entry.label, entry.spec) for entry in list_voices()]


def _voice_dropdown_update(selected: str | None = None):
    choices = _voice_choices()
    specs = [spec for _, spec in choices]
    value = selected if selected in specs else (specs[0] if specs else None)
    return gr.update(choices=choices, value=value)


# ---------------------------------------------------------------- design tab


def _generate_design(instruct: str, ref_text: str):
    """Render a candidate voice into memory, saved only if it earns a name.

    Voice design is a lottery ticket — the same persona renders differently
    each run — so the natural loop is generate, listen, generate again.
    Nothing touches ``voices/`` until :func:`_save_design`.
    """

    if not instruct.strip() or not ref_text.strip():
        yield "Both a persona and narration text are required.", None, None
        return

    def work() -> dict:
        with gpu_slot():
            provider = synthesis_provider()
            print("Designing a candidate voice...")
            clip = provider.design(persona=instruct, ref_text=ref_text, language=LANGUAGE)
        print(f"Rendered {len(clip.audio) / clip.sample_rate:.1f}s. Listen, then save or retry.")
        return {
            "audio": clip.audio,
            "sample_rate": int(clip.sample_rate),
            "instruct": instruct,
            "ref_text": ref_text,
            "design_model": loaded_model_name(),
        }

    for log, result in stream_output(work):
        if result is None:
            yield log, None, None
        else:
            yield log, (result["sample_rate"], result["audio"]), result


def _save_design(pending: dict | None, name: str):
    """Commit the pending candidate to ``voices/<name>/``."""

    if not pending:
        return "Generate a voice first.", gr.update(), gr.update()
    name = name.strip().replace("/", "_")
    if not name:
        return "Give the voice a name.", gr.update(), gr.update()

    voice_dir = VOICES_DIR / name
    existed = voice_dir.exists()
    voice_dir.mkdir(parents=True, exist_ok=True)
    sf.write(
        voice_dir / VOICE_REFERENCE_AUDIO_FILENAME,
        pending["audio"],
        pending["sample_rate"],
    )
    (voice_dir / VOICE_REFERENCE_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "slug": name,
                "instruct": pending["instruct"],
                "ref_text": pending["ref_text"],
                "sample_rate": pending["sample_rate"],
                "design_model": pending.get("design_model"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    note = " (replaced the previous voice of that name)" if existed else ""
    return (
        f"Saved to {voice_dir}{note}.",
        _voice_dropdown_update(name),
        _voice_dropdown_update(name),
    )


def _discard_design():
    return "Discarded.", None, None


# ----------------------------------------------------------------- clone tab


def _mode_note(ref_text: str) -> str:
    """Tell the user what the current transcript means for the clone."""

    if ref_text.strip():
        return (
            "Clones **timbre + prosody**. Verify the transcript word for word "
            "against what was actually said — a misheard word teaches the clone "
            "a wrong alignment. Clear it to clone timbre only."
        )
    return "No transcript — clones **timbre only**, in the model's own cadence."


def _process_recording(upload: str | None):
    """Condition an uploaded recording and recover its transcript, unsaved.

    The recording is staged outside ``voices/`` so nothing exists until it is
    auditioned and saved — the same commit-last loop as the design tab.  What
    comes back is what cloning will actually use: the trimmed, levelled audio
    and the ASR transcript, both open to inspection before anything is kept.
    """

    if not upload:
        yield "Record or choose an audio file first.", None, "", "", None
        return

    def work() -> dict:
        source = Path(upload)
        staged = Path(tempfile.mkdtemp(prefix="voice-staging-")) / source.name
        shutil.copyfile(source, staged)
        with gpu_slot():  # ASR runs on the GPU
            voice = resolve_voice(str(staged), transcribe_missing=REFERENCE_TRANSCRIBE)
        print(describe(voice))
        return {
            "path": str(staged),
            "ref_text": voice.ref_text or "",
            "sample_rate": voice.sample_rate,
            "audio": voice.audio,
        }

    for log, result in stream_output(work):
        if result is None:
            yield log, None, "", "", None
        else:
            yield (
                log,
                (result["sample_rate"], result["audio"]),
                result["ref_text"],
                _mode_note(result["ref_text"]),
                result,
            )


def _audition_pending(pending: dict | None, transcript: str, text: str):
    """Clone a passage from the staged recording, before it is saved."""

    if not pending:
        yield "Process a recording first.", None
        return
    yield from _render_with_voice(pending["path"], transcript.strip() or None, text)


def _save_clone(pending: dict | None, transcript: str, name: str):
    """Commit the staged recording (and its verified transcript) to ``voices/``."""

    if not pending:
        return "Process a recording first.", gr.update(), gr.update()

    transcript = transcript.strip()
    spec = import_recording(pending["path"], name, transcript)
    mode = "timbre + prosody" if transcript else "timbre only"
    return (
        f"Saved {VOICES_DIR / spec} — clones {mode}.",
        _voice_dropdown_update(spec),
        _voice_dropdown_update(spec),
    )


def _discard_clone():
    return "Discarded.", None, "", "", None


def _render_with_voice(spec: str, ref_text: str | None, text: str):
    """Shared audition loop: yields the log, then the rendered passage.

    ``load_voice`` hides what the spec names — a designed clip, a recording,
    or a speaker the backend carries itself — so auditioning exercises
    exactly the narration path a book run would take.
    """

    if not text.strip():
        yield "Enter something for the voice to read.", None
        return

    def work() -> tuple[int, np.ndarray]:
        with gpu_slot():
            provider = synthesis_provider()
            voice = provider.load_voice(spec, ref_text=ref_text)
            print("Rendering passage...")
            clip = provider.generate(
                text=text, language=LANGUAGE, voice=voice, instruction=NARRATION_INSTRUCTION
            )
        return int(clip.sample_rate), clip.audio

    for log, result in stream_output(work):
        yield log, result


def _voice_detail(spec: str | None):
    """Populate the inspector for the selected voice."""

    entry = find_voice(spec) if spec else None
    if entry is None:
        return None, "", "Select a voice."

    if entry.builtin:
        summary = (
            f"**{entry.spec}** — built-in speaker of the synthesis backend\n\n"
            "Lives in the model checkpoint, so there is no reference clip or "
            "transcript to inspect. Narration renders it natively at full "
            "quality — audition below to hear it."
        )
        return None, "", summary

    mode = "timbre + prosody" if entry.has_transcript else "timbre only (no transcript)"
    summary = (
        f"**{entry.spec}** — clones {mode}\n\n"
        f"Audio: `{entry.audio_path}`\n\n"
        f"Transcript: `{entry.transcript_path}`"
        f"{'' if entry.transcript_path.exists() else ' *(not written yet)*'}"
    )
    return str(entry.audio_path), entry.ref_text or "", summary


def _save_transcript(spec: str | None, text: str) -> str:
    entry = find_voice(spec) if spec else None
    if entry is None:
        return "Select a voice first."
    try:
        return save_transcript(entry, text)
    except ValueError as exc:
        return str(exc)


def _retranscribe(spec: str | None, current: str):
    """Recognise the reference audio again, into the box rather than to disk.

    Useful after editing a transcript into something worse, or when a voice was
    saved with a hand-written transcript that may not match the recording.  The
    result replaces the textbox contents only — saving stays a separate,
    deliberate click, so a fresh recognition can be compared against what is
    already there and rejected.
    """

    if not spec:
        yield "Select a voice first.", gr.update()
        return
    entry = find_voice(spec)
    if entry is not None and entry.builtin:
        yield "Built-in speakers have no reference recording to transcribe.", gr.update()
        return

    def work() -> str | None:
        from ..synthesis.transcribe import transcribe

        with gpu_slot():
            # Transcribe exactly what cloning conditions on: trimmed and
            # levelled, not the raw file.
            voice = resolve_voice(spec, transcribe_missing=False)
            print(f"Re-transcribing {len(voice.audio) / voice.sample_rate:.1f}s...")
            return transcribe(voice.audio, voice.sample_rate)

    for log, result in stream_output(work):
        if result is None:
            # Also the terminal state when recognition was rejected as too thin;
            # the log says so, and the existing transcript is left alone.
            yield log, gr.update()
        else:
            verdict = (
                "differs from the box — review it before saving"
                if result.strip() != current.strip()
                else "matches the box"
            )
            yield f"{log}\nRecognised text {verdict}. Nothing saved yet.", result


def _audition(spec: str | None, text: str):
    """Render a passage with a saved voice so it can be judged by ear.

    Whatever kind of voice the spec names, the backend resolves it the same
    way narration will, so what is heard here is what a book run produces.
    """

    if not spec:
        yield "Select a voice first.", None
        return
    # ref_text=None lets a file-backed voice keep its own transcript.
    yield from _render_with_voice(spec, None, text)


def _active_voice_warning(spec: str) -> str:
    """Renaming or deleting the CLI's default voice deserves a heads-up."""

    if spec == str(ACTIVE_VOICE):
        return (
            " **Note:** CLI runs default to this voice — update ACTIVE_VOICE "
            "in src/audiobook/config.py."
        )
    return ""


def _voice_display_name(spec: str) -> str:
    """The editable part of a spec: a folder name, or a recording's stem."""

    entry = find_voice(spec)
    if entry is None or entry.builtin:
        return ""
    return entry.spec if entry.folder else Path(entry.audio_path).stem


def _begin_rename(spec: str | None):
    """Open the rename editor, seeded with the name being changed."""

    if not spec:
        return (
            gr.update(visible=False),
            gr.update(),
            gr.update(visible=False),
            "Select a voice first.",
        )
    return (
        gr.update(visible=True),
        gr.update(value=_voice_display_name(spec)),
        gr.update(visible=False),  # rename and delete are mutually exclusive
        "",
    )


def _begin_delete(spec: str | None):
    """Open the delete confirmation, naming exactly what would be removed."""

    if not spec:
        return (
            gr.update(visible=False),
            gr.update(),
            gr.update(visible=False),
            "Select a voice first.",
        )
    return (
        gr.update(visible=True),
        gr.update(value=f"Delete **{spec}**? This cannot be undone."),
        gr.update(visible=False),
        "",
    )


def _cancel_edits():
    """Close both inline editors."""

    return gr.update(visible=False), gr.update(visible=False), ""


def _rename_voice(spec: str | None, new_name: str):
    """Rename, closing the editor only when the rename actually succeeded."""

    entry = find_voice(spec) if spec else None
    if entry is None:
        return "Select a voice first.", gr.update(), gr.update(), gr.update()
    try:
        new_spec = rename_voice(entry, new_name)
    except (ValueError, OSError) as exc:
        # Leave the editor open so the rejected name can be corrected in place.
        return str(exc), gr.update(), gr.update(), gr.update(visible=True)
    return (
        f"Renamed to {new_spec}.{_active_voice_warning(spec)}",
        _voice_dropdown_update(new_spec),
        _voice_dropdown_update(new_spec),
        gr.update(visible=False),
    )


def _confirm_delete(spec: str | None):
    """Delete the voice; the two-step reveal is the confirmation."""

    entry = find_voice(spec) if spec else None
    if entry is None:
        return "Select a voice first.", gr.update(), gr.update(), gr.update(visible=False)
    try:
        message = delete_voice(entry)
    except (ValueError, OSError) as exc:
        return str(exc), gr.update(), gr.update(), gr.update(visible=False)
    return (
        f"{message}{_active_voice_warning(spec)}",
        _voice_dropdown_update(),
        _voice_dropdown_update(),
        gr.update(visible=False),
    )


# ------------------------------------------------------------- narration tab


def _default_descriptor():
    return provider_descriptor(DEFAULT_PREPARATION_PROVIDER)


def _provider_choices() -> list[tuple[str, str]]:
    """Every registered provider, labelled as its adapter describes itself."""

    return [(descriptor.label, descriptor.name) for descriptor in provider_descriptors()]


def _model_dropdown_update(provider: str | None):
    """Repoint the model dropdown at the chosen provider's declared models."""

    try:
        descriptor = provider_descriptor(provider or DEFAULT_PREPARATION_PROVIDER)
    except ValueError:
        return gr.update(choices=[], value=None), ""
    return (
        gr.update(choices=list(descriptor.models), value=descriptor.default_model),
        _provider_note(descriptor),
    )


def _provider_note(descriptor) -> str:
    """Where the work goes, and anything stopping it from going there.

    Base URL is shown but not editable: it is configuration that is either
    right or wrong for the whole machine, not a per-run decision.
    """

    where = descriptor.base_url or "provider default endpoint"
    origin = "runs locally" if descriptor.local else "sends book text to a third party"
    missing = descriptor.missing_requirement()
    note = f"{where} — {origin}."
    if missing:
        note += f" **Unavailable: {missing}**"
    return note


def _default_book_path() -> str | None:
    """Preselect a book sitting beside the project, whatever format it is in.

    DEFAULT_BOOK_PATH names one file, but the picker accepts every supported
    format, so a ``book.epub`` next to the configured ``book.pdf`` is just as
    good a starting point as the one that happens to be spelled in config.
    """

    candidates = [DEFAULT_BOOK_PATH] + [
        DEFAULT_BOOK_PATH.with_suffix(suffix) for suffix in SUPPORTED_SOURCE_SUFFIXES
    ]
    return next((str(path) for path in candidates if path.exists()), None)


def _prepare(
    book: str | None,
    output_dir: str,
    provider: str,
    model: str,
    timeout: float,
    preview_chapters: int,
    preview_units: int,
    force: bool,
):
    """Extract and adapt a book, then show the reviewable markdown.

    Base URL and credentials are deliberately not exposed: they describe where
    the LLM lives, which is either right in config or wrong everywhere, and
    preflight already reports which it is.  Provider and model are real
    choices, taken from what the adapters declare.
    """

    if not book:
        yield "Choose a book first.", gr.update()
        return

    try:
        descriptor = provider_descriptor(provider or DEFAULT_PREPARATION_PROVIDER)
    except ValueError as exc:
        yield str(exc), gr.update()
        return
    # Catching this here saves the minutes of extraction that precede the
    # first LLM call, which is the whole point of asking providers to declare.
    missing = descriptor.missing_requirement()
    if missing:
        yield f"{descriptor.label} is not usable: {missing}", gr.update()
        return

    options = PreparationWorkflowOptions(
        source_path=Path(book),
        output_dir=Path(output_dir),
        provider_name=descriptor.name,
        model=model or descriptor.default_model,
        base_url=descriptor.base_url or "",
        timeout_seconds=float(timeout),
        # 0 means "all" in the UI; the pipeline expects None or a positive count.
        preview_chapters=int(preview_chapters) or None,
        preview_units=int(preview_units) or None,
        force=force,
    )

    for log, result in stream_output(lambda: prepare_narration_script(options)):
        if result is None:
            yield log, gr.update()
        else:
            # Selecting the artifact is all this has to do: the Review panels
            # hang off the picker, so they fill in the same way whether a run
            # just produced the book or it was chosen from disk.
            script_path = resolve_script_path(options.output_dir, options.script_path)
            yield log, _script_dropdown_update(str(script_path))


def _script_dropdown_update(selected: str | None = None):
    choices = [str(path) for path in list_prepared_scripts()]
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _load_review(script: str | None, selected_model: str | None = None):
    """Fill every Review panel for the artifact chosen in the dropdown.

    Returns the header, the list of units needing attention, the first of
    those rendered as a change, the full prepared text, and the units keyed by
    id for the detail panel.  Reading the prose is the fallback view: what a
    reviewer needs first is what was altered.
    """

    if not script:
        return "", gr.update(choices=[], value=None), "", None, {}

    summary = summarize(script, selected_model)
    try:
        book = load_artifact(script)
    except Exception:
        # summarize() already rendered the reason; the panels stay empty.
        return summary, gr.update(choices=[], value=None), "", None, {}

    units = flagged_units(book)
    keyed = {unit.unit_id: unit for unit in units}
    choices = [(unit.label, unit.unit_id) for unit in units]
    first = units[0] if units else None
    detail = render_unit(first) if first else "_The model changed nothing and flagged nothing._"

    # Older artifacts predate the markdown companion file; write it on demand
    # so the download button always has a real file to point at.
    markdown = prepared_markdown_path(Path(script))
    if not markdown.exists():
        write_prepared_markdown(book, markdown)
    return (
        summary,
        gr.update(choices=choices, value=first.unit_id if first else None),
        detail,
        str(markdown),
        keyed,
    )


def _show_unit(unit_id: str | None, keyed: dict) -> str:
    return render_unit((keyed or {}).get(unit_id) if unit_id else None)


def _narrate(
    script: str | None,
    voice: str | None,
    output_dir: str,
    preview_chunks: int,
    dry_run: bool,
):
    """Narrate a prepared script, as a preview or in full."""

    if not script:
        yield "Prepare or select a script first.", None, gr.update()
        return

    options = NarrationWorkflowOptions(
        output_dir=Path(output_dir),
        tts_model=str(LOCAL_TTS_MODEL_PATH if LOCAL_TTS_MODEL_PATH.exists() else TTS_MODEL),
        script_path=Path(script),
        preview_chunks=int(preview_chunks) or None,
        dry_run=dry_run,
        voice=voice,
    )

    def work():
        # Narration loads its own checkpoint, so free the audition model first
        # rather than discovering the collision as an out-of-memory error.
        with gpu_slot():
            unload_model()
            return narrate_prepared_script(options)

    for log, result in stream_output(work):
        if result is None:
            yield log, None, gr.update()
        else:
            yield log, str(result), _audiobook_update()


def _audiobook_update(selected: str | None = None):
    choices = [str(path) for path in list_audiobooks()]
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _gpu_status() -> str:
    name = loaded_model_name()
    return f"GPU: {name}" if name else "GPU: idle"


# ------------------------------------------------------------------- the app


def build_app() -> gr.Blocks:
    """Assemble the four-tab interface.

    Tabs follow the backend's declared capabilities: a provider that cannot
    design, clone, or narrate simply does not show that tab, so swapping in a
    partial backend narrows the app instead of breaking it.
    """

    capabilities = synthesis_descriptor(DEFAULT_SYNTHESIS_PROVIDER)

    with gr.Blocks(title="Audiobook Studio") as app:
        gr.Markdown("# Audiobook Studio")
        with gr.Row():
            gpu_status = gr.Markdown(_gpu_status())
            free_gpu = gr.Button("Free GPU", scale=0)

        with gr.Tab("Design a voice", visible=capabilities.supports_design):
            gr.Markdown(
                "Describe a narrator and give them something to read. Generate as "
                "many candidates as you like — nothing is kept until you save one "
                "with a name."
            )
            design_instruct = gr.Textbox(
                label="Narrator persona", value=VOICE_DESIGN_INSTRUCT, lines=6
            )
            design_ref_text = gr.Textbox(
                label="Narration text (what the voice reads)",
                value=VOICE_REFERENCE_TEXT,
                lines=4,
            )
            design_generate = gr.Button("Generate", variant="primary")
            design_audio = gr.Audio(label="Candidate", type="numpy")
            with gr.Row():
                design_name = gr.Textbox(label="Save as", placeholder="warm_male_v3", scale=2)
                design_save = gr.Button("Save voice", scale=1)
                design_discard = gr.Button("Discard", scale=1)
            design_status = gr.Markdown()
            design_log = gr.Textbox(label="Log", lines=6, max_lines=16)
            design_pending = gr.State(None)

        with gr.Tab("Clone a voice", visible=capabilities.supports_clone):
            gr.Markdown(
                "Clone a real voice from a recording. Aim for 15–20 seconds of "
                "plain narration, recorded close to the microphone in a quiet "
                "room. Nothing is kept until you save it with a name."
            )
            with gr.Accordion("Example script to read", open=False):
                gr.Markdown(
                    f"> {VOICE_REFERENCE_TEXT}\n\n"
                    "Read it at your natural narration pace — the clone copies "
                    "how you read, not just how you sound. Reading a known "
                    "script also makes the recognised transcript easy to check "
                    "word for word."
                )
            import_file = gr.Audio(
                label="Recording", type="filepath", sources=["upload", "microphone"]
            )
            process_button = gr.Button("Process recording", variant="primary")
            processed_audio = gr.Audio(
                label="What the clone will hear (trimmed and levelled)",
                type="numpy",
            )
            clone_transcript = gr.Textbox(
                label="Transcript (recognised automatically — fix any errors)",
                lines=4,
            )
            clone_mode = gr.Markdown()
            with gr.Row():
                clone_audition_text = gr.Textbox(
                    label="Audition passage", value=AUDITION_TEXT, lines=2, scale=3
                )
                clone_audition_button = gr.Button("Audition", scale=1)
            clone_audition_audio = gr.Audio(label="Cloned result", type="numpy")
            with gr.Row():
                clone_name = gr.Textbox(
                    label="Save as",
                    placeholder="Defaults to the recording's filename",
                    scale=2,
                )
                clone_save = gr.Button("Save voice", scale=1)
                clone_discard = gr.Button("Discard", scale=1)
            clone_status = gr.Markdown()
            clone_log = gr.Textbox(label="Log", lines=6, max_lines=16)
            clone_pending = gr.State(None)

        with gr.Tab("Book narration", visible=capabilities.supports_narrate):
            gr.Markdown(
                "Book in, audiobook out — in two steps with a review between. "
                "Prepare adapts the text for listening; read the result before "
                "spending GPU-hours narrating it."
            )

            gr.Markdown("## 1 · Prepare")
            with gr.Row():
                book_input = gr.File(
                    label="Book (PDF or EPUB)",
                    file_types=list(SUPPORTED_SOURCE_SUFFIXES),
                    type="filepath",
                    value=_default_book_path(),
                )
                prepare_output_dir = gr.Textbox(
                    label="Output directory", value=str(DEFAULT_OUTPUT_DIR)
                )
            with gr.Accordion("Preparation settings", open=False):
                with gr.Row():
                    provider = gr.Dropdown(
                        choices=_provider_choices(),
                        value=DEFAULT_PREPARATION_PROVIDER,
                        label="Provider",
                        interactive=True,
                    )
                    preparation_model = gr.Dropdown(
                        choices=list(_default_descriptor().models),
                        value=DEFAULT_PREPARATION_MODEL,
                        label="Model",
                        interactive=True,
                    )
                    timeout = gr.Number(label="Timeout (s)", value=DEFAULT_PROVIDER_TIMEOUT_SECONDS)
                provider_note = gr.Markdown(_provider_note(_default_descriptor()))
                with gr.Row():
                    prep_preview_chapters = gr.Number(
                        label="First N chapters (0 = all)", value=0, precision=0
                    )
                    prep_preview_units = gr.Number(
                        label="First N units (0 = all)", value=0, precision=0
                    )
                    force_preparation = gr.Checkbox(label="Ignore cached units", value=False)
            prepare_button = gr.Button("Prepare", variant="primary")
            prepare_log = gr.Textbox(label="Log", lines=8, max_lines=20)

            gr.Markdown("## 2 · Review")
            with gr.Row():
                script_picker = gr.Dropdown(
                    choices=[str(p) for p in list_prepared_scripts()],
                    label="Prepared script",
                    interactive=True,
                    scale=3,
                )
                refresh_scripts = gr.Button("Refresh", scale=0)
            # The header answers "is this artifact worth reviewing" — which
            # model wrote it, whether the run finished, whether the PDF still
            # matches — before any of its prose is read.
            review_summary = gr.Markdown()
            with gr.Tabs():
                # Changes first: a book is reviewed by its edits, not by
                # re-reading the source it was made from.
                with gr.Tab("Changes"):
                    flagged_picker = gr.Dropdown(
                        label="Units the model changed or flagged",
                        interactive=True,
                    )
                    flagged_detail = gr.Markdown()
                with gr.Tab("Full text"):
                    # The prepared book can run long; offer it as a file
                    # rather than rendering the whole thing inline.
                    review = gr.DownloadButton(label="Download full text")
            flagged_state = gr.State({})

            gr.Markdown("## 3 · Narrate")
            with gr.Row():
                narrate_voice = gr.Dropdown(
                    choices=_voice_choices(), label="Voice", interactive=True
                )
                narrate_output_dir = gr.Textbox(
                    label="Output directory", value=str(DEFAULT_OUTPUT_DIR)
                )
                narrate_preview_chunks = gr.Number(
                    label="Preview: first N chunks (0 = whole book)",
                    value=3,
                    precision=0,
                )
                dry_run = gr.Checkbox(label="Plan only (no audio)", value=False)
            narrate_button = gr.Button("Narrate", variant="primary")
            narrate_log = gr.Textbox(label="Log", lines=10, max_lines=25)
            narrate_result = gr.Audio(label="Result", type="filepath")

        with gr.Tab("Library"):
            gr.Markdown("## Voices")
            # Picker and its inline editors share a Group so they read as one
            # card. The action buttons sit on their own row beneath the
            # dropdown: putting buttons beside a labelled input misaligns them
            # against the label, which is what made the first layout a mess.
            with gr.Group():
                voice_picker = gr.Dropdown(
                    choices=_voice_choices(), label="Voice", interactive=True
                )
                with gr.Row():
                    rename_open = gr.Button("✏️ Rename", size="sm")
                    delete_open = gr.Button("🗑️ Delete", size="sm")
                    refresh_voices = gr.Button("⟳ Refresh", size="sm")
                with gr.Row(visible=False) as rename_row:
                    rename_input = gr.Textbox(placeholder="New name", container=False, scale=3)
                    rename_button = gr.Button("Save", variant="primary", scale=1)
                    rename_cancel = gr.Button("Cancel", scale=1)
                with gr.Row(visible=False) as delete_row:
                    delete_warning = gr.Markdown(scale=3)
                    delete_button = gr.Button("Confirm delete", variant="stop", scale=1)
                    delete_cancel = gr.Button("Cancel", scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    voice_summary = gr.Markdown("Select a voice.")
                    voice_audio = gr.Audio(label="Reference clip", type="filepath")
                with gr.Column(scale=1):
                    voice_transcript = gr.Textbox(
                        label="Transcript (must match the audio word for word)",
                        lines=6,
                    )
                    save_transcript_button = gr.Button("Save transcript")
                    retranscribe_button = gr.Button("Infer transcript again")

            with gr.Row():
                audition_text = gr.Textbox(
                    label="Audition passage", value=AUDITION_TEXT, lines=2, scale=3
                )
                audition_button = gr.Button("Audition", scale=1)
            audition_audio = gr.Audio(label="Result", type="numpy")

            voice_status = gr.Markdown()
            library_log = gr.Textbox(label="Log", lines=6, max_lines=16)

            gr.Markdown("## Audiobooks")
            with gr.Row():
                audiobook_picker = gr.Dropdown(
                    choices=[str(p) for p in list_audiobooks()],
                    label="Audiobook",
                    interactive=True,
                    scale=3,
                )
                refresh_audiobooks = gr.Button("Refresh", scale=0)
            audiobook_player = gr.Audio(label="Play", type="filepath")
            audiobook_download = gr.File(label="Download")

        # -------------------------------------------------------- wiring

        # Every heavy action is wired begin → work → end.  begin and end run
        # with queue=False so they land immediately rather than queueing behind
        # the very run they exist to announce; concurrency_id groups the work
        # itself, so a second browser tab queues visibly instead of blocking on
        # the lock; trigger_mode="once" is the double-click guard.
        heavy_buttons = [
            free_gpu,
            design_generate,
            process_button,
            clone_audition_button,
            retranscribe_button,
            audition_button,
            prepare_button,
            narrate_button,
        ]
        assert len(heavy_buttons) == len(HEAVY_ACTIONS)
        busy = dict(outputs=heavy_buttons, queue=False, show_progress="hidden")
        release = dict(fn=_end_work, **busy)

        free_gpu.click(_begin_work(0), **busy).then(
            lambda: (unload_model(), _gpu_status())[1],
            outputs=gpu_status,
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release)

        design_generate.click(
            _begin_work(1, None, None),
            outputs=heavy_buttons + [design_audio, design_pending],
            queue=False,
            show_progress="hidden",
        ).then(
            _generate_design,
            inputs=[design_instruct, design_ref_text],
            outputs=[design_log, design_audio, design_pending],
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release).then(_gpu_status, outputs=gpu_status)
        design_save.click(
            _save_design,
            inputs=[design_pending, design_name],
            outputs=[design_status, voice_picker, narrate_voice],
        )
        design_discard.click(_discard_design, outputs=[design_status, design_audio, design_pending])

        process_button.click(
            _begin_work(2, None, "", "", None),
            outputs=heavy_buttons + [processed_audio, clone_transcript, clone_mode, clone_pending],
            queue=False,
            show_progress="hidden",
        ).then(
            _process_recording,
            inputs=import_file,
            outputs=[
                clone_log,
                processed_audio,
                clone_transcript,
                clone_mode,
                clone_pending,
            ],
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release).then(_gpu_status, outputs=gpu_status)
        clone_transcript.change(_mode_note, inputs=clone_transcript, outputs=clone_mode)
        clone_audition_button.click(
            _begin_work(3, None),
            outputs=heavy_buttons + [clone_audition_audio],
            queue=False,
            show_progress="hidden",
        ).then(
            _audition_pending,
            inputs=[clone_pending, clone_transcript, clone_audition_text],
            outputs=[clone_log, clone_audition_audio],
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release).then(_gpu_status, outputs=gpu_status)
        clone_save.click(
            _save_clone,
            inputs=[clone_pending, clone_transcript, clone_name],
            outputs=[clone_status, voice_picker, narrate_voice],
        )
        clone_discard.click(
            _discard_clone,
            outputs=[
                clone_status,
                processed_audio,
                clone_transcript,
                clone_mode,
                clone_pending,
            ],
        )
        voice_picker.change(
            _voice_detail,
            inputs=voice_picker,
            outputs=[voice_audio, voice_transcript, voice_summary],
        )
        refresh_voices.click(_voice_dropdown_update, outputs=voice_picker)
        save_transcript_button.click(
            _save_transcript,
            inputs=[voice_picker, voice_transcript],
            outputs=voice_status,
        )
        # The transcript box is deliberately not cleared: _retranscribe compares
        # the new recognition against what is in it, which is the whole point.
        retranscribe_button.click(_begin_work(4), **busy).then(
            _retranscribe,
            inputs=[voice_picker, voice_transcript],
            outputs=[library_log, voice_transcript],
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release).then(_gpu_status, outputs=gpu_status)
        audition_button.click(
            _begin_work(5, None),
            outputs=heavy_buttons + [audition_audio],
            queue=False,
            show_progress="hidden",
        ).then(
            _audition,
            inputs=[voice_picker, audition_text],
            outputs=[library_log, audition_audio],
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release).then(_gpu_status, outputs=gpu_status)
        rename_open.click(
            _begin_rename,
            inputs=voice_picker,
            outputs=[rename_row, rename_input, delete_row, voice_status],
        )
        delete_open.click(
            _begin_delete,
            inputs=voice_picker,
            outputs=[delete_row, delete_warning, rename_row, voice_status],
        )
        rename_cancel.click(_cancel_edits, outputs=[rename_row, delete_row, voice_status])
        delete_cancel.click(_cancel_edits, outputs=[rename_row, delete_row, voice_status])
        rename_button.click(
            _rename_voice,
            inputs=[voice_picker, rename_input],
            outputs=[voice_status, voice_picker, narrate_voice, rename_row],
        )
        delete_button.click(
            _confirm_delete,
            inputs=voice_picker,
            outputs=[voice_status, voice_picker, narrate_voice, delete_row],
        )
        # Switching voices mid-edit would otherwise act on the wrong one.
        voice_picker.change(_cancel_edits, outputs=[rename_row, delete_row, voice_status])

        prepare_button.click(
            _begin_work(
                6,
                "_Preparing a new artifact..._",
                gr.update(choices=[], value=None),
                "",
                None,
                {},
            ),
            outputs=heavy_buttons
            + [review_summary, flagged_picker, flagged_detail, review, flagged_state],
            queue=False,
            show_progress="hidden",
        ).then(
            _prepare,
            inputs=[
                book_input,
                prepare_output_dir,
                provider,
                preparation_model,
                timeout,
                prep_preview_chapters,
                prep_preview_units,
                force_preparation,
            ],
            outputs=[prepare_log, script_picker],
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release).then(
            _load_review,
            inputs=[script_picker, preparation_model],
            outputs=[
                review_summary,
                flagged_picker,
                flagged_detail,
                review,
                flagged_state,
            ],
        )
        # The model list belongs to the provider; changing one must repoint the
        # other, or a run would ask Ollama for a hosted model.
        provider.change(
            _model_dropdown_update,
            inputs=provider,
            outputs=[preparation_model, provider_note],
        )
        refresh_scripts.click(_script_dropdown_update, outputs=script_picker)
        # Reading an artifact parses a large JSON and hashes the source PDF, so
        # say so first rather than leaving the old book's header standing.
        script_picker.change(
            lambda: "_Reading artifact..._",
            outputs=review_summary,
            queue=False,
            show_progress="hidden",
        ).then(
            _load_review,
            inputs=[script_picker, preparation_model],
            outputs=[
                review_summary,
                flagged_picker,
                flagged_detail,
                review,
                flagged_state,
            ],
        )
        flagged_picker.change(
            _show_unit, inputs=[flagged_picker, flagged_state], outputs=flagged_detail
        )

        narrate_button.click(
            _begin_work(7, None),
            outputs=heavy_buttons + [narrate_result],
            queue=False,
            show_progress="hidden",
        ).then(
            _narrate,
            inputs=[
                script_picker,
                narrate_voice,
                narrate_output_dir,
                narrate_preview_chunks,
                dry_run,
            ],
            outputs=[narrate_log, narrate_result, audiobook_picker],
            concurrency_id="heavy",
            trigger_mode="once",
        ).then(**release).then(_gpu_status, outputs=gpu_status)

        refresh_audiobooks.click(_audiobook_update, outputs=audiobook_picker)
        audiobook_picker.change(
            lambda path: (path, path),
            inputs=audiobook_picker,
            outputs=[audiobook_player, audiobook_download],
        )

        app.load(_voice_dropdown_update, outputs=voice_picker)
        app.load(_voice_dropdown_update, outputs=narrate_voice)

    return app


def launch(**kwargs) -> None:
    """Serve the app on localhost."""

    build_app().queue(default_concurrency_limit=1).launch(**kwargs)


__all__ = ["build_app", "launch"]
