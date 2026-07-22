"""Startup checks: probe everything the configured pipeline needs, up front.

Each stage already verifies its own dependencies, but only at the moment it
needs them — ffmpeg is checked after TTS has generated audio, the Ollama model
after extraction has run.  Failing there wastes the minutes of work that came
before.  This module asks the same questions before any work starts, reading
the answers from the same config the stages will use.

Checks are ordered from environment to configuration, and none of them loads a
model or decodes audio: a preflight with everything in place costs seconds.
The exception is a missing model — the Ollama model, a TTS checkpoint, the
Whisper ASR model — which is downloaded here, with progress shown: waiting for
the download up front beats a run stalling on it after extraction already ran.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import (
    ACTIVE_VOICE,
    ASR_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREPARATION_MODEL,
    DEFAULT_PREPARATION_PROVIDER,
    DEFAULT_PROVIDER_BASE_URL,
    DEFAULT_SYNTHESIS_PROVIDER,
    LOCAL_TTS_MODEL_PATH,
    LOCAL_VOICE_CLONE_MODEL_PATH,
    LOCAL_VOICE_DESIGN_MODEL_PATH,
    REFERENCE_TRANSCRIBE,
    TTS_MODEL,
    VOICE_CLONE_MODEL,
    VOICE_DESIGN_MODEL,
    VOICES_DIR,
)
from .synthesis.voices import AUDIO_SUFFIXES

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one probe: what was checked, how it went, and what to do."""

    name: str
    status: str
    detail: str


def _check_ffmpeg() -> CheckResult:
    path = shutil.which("ffmpeg")
    if path:
        return CheckResult("ffmpeg", OK, path)
    return CheckResult(
        "ffmpeg",
        FAIL,
        "Not on PATH. Needed to decode reference voices and write the M4B. "
        "Install it (e.g. apt install ffmpeg).",
    )


def _check_packages() -> CheckResult:
    import importlib.util

    required = ["torch", "qwen_tts", "soundfile", "numpy", "tqdm", "pymupdf4llm"]
    if REFERENCE_TRANSCRIBE:
        required += ["librosa", "transformers"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return CheckResult("python packages", OK, ", ".join(required))
    return CheckResult(
        "python packages",
        FAIL,
        f"Missing: {', '.join(missing)}. Run: .venv/bin/python -m pip install -r requirements.txt",
    )


def _check_cuda() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("CUDA GPU", FAIL, "torch is not installed.")
    if not torch.cuda.is_available():
        return CheckResult(
            "CUDA GPU",
            FAIL,
            "No CUDA device. TTS synthesis, voice design and cloning all require one.",
        )
    properties = torch.cuda.get_device_properties(0)
    total_gb = properties.total_memory / 1024**3
    detail = f"{properties.name}, {total_gb:.0f} GB"
    if total_gb < 6:
        return CheckResult(
            "CUDA GPU",
            WARN,
            f"{detail} — under 6 GB; a 1.7B checkpoint plus generation may not fit.",
        )
    return CheckResult("CUDA GPU", OK, detail)


def _download_checkpoint(local_path: Path, remote_id: str) -> None:
    """Fetch the full snapshot of ``remote_id`` into ``local_path``, atomically.

    huggingface_hub renders a tqdm bar per file while it downloads.  The
    snapshot lands in a sibling ``.partial`` directory and is renamed only once
    complete, so an interrupted download cannot leave a half-written checkpoint
    that the next preflight's ``local_path.exists()`` would take for a real
    one.  The partial directory keeps huggingface_hub's resume metadata, so a
    retried preflight continues the download instead of restarting it.
    """

    from huggingface_hub import snapshot_download

    partial = local_path.with_name(local_path.name + ".partial")
    snapshot_download(remote_id, local_dir=partial)
    # Resume metadata has served its purpose; the checkpoint itself is complete.
    shutil.rmtree(partial / ".cache", ignore_errors=True)
    partial.rename(local_path)


def _check_checkpoint(name: str, local_path: Path, remote_id: str) -> CheckResult:
    """A missing local checkpoint is downloaded now rather than mid-run.

    Synthesis would fetch it on first use anyway, but that download lands in
    the Hugging Face cache instead of ``local_path`` — so the checkpoint would
    read as missing on every later preflight — and it stalls the run right
    when the first chapter is about to speak.  A failed download is a warning,
    not an error, because that lazy first-use path still remains.
    """

    if local_path.exists():
        return CheckResult(name, OK, str(local_path))
    print(f"  {name}: downloading {remote_id} to {local_path}", flush=True)
    try:
        _download_checkpoint(local_path, remote_id)
    except Exception as exc:  # noqa: BLE001 - any failure means "not downloaded"
        return CheckResult(
            name,
            WARN,
            f"{local_path} not found and downloading {remote_id} failed "
            f"({exc}); will retry from Hugging Face on first use.",
        )
    return CheckResult(name, OK, f"{local_path} — downloaded on this run")


def _active_voice_info():
    """The catalog row ACTIVE_VOICE names, or ``None`` if the backend lists
    nothing by that spec.

    Cheap by contract: ``voices()`` reads metadata files and never loads a
    model, which is exactly what preflight is allowed to do.
    """

    from .synthesis.providers import create_synthesis_provider

    spec = str(ACTIVE_VOICE)
    for info in create_synthesis_provider(DEFAULT_SYNTHESIS_PROVIDER).voices():
        if info.spec == spec or (info.builtin and info.spec.casefold() == spec.casefold()):
            return info
    return None


def _checkpoint_checks() -> list[CheckResult]:
    """The checkpoints the active voice actually needs.

    A voice the backend carries itself renders on the CustomVoice checkpoint;
    a file-backed voice clones on the Base checkpoint, with the Design
    checkpoint alongside it so new voices can be made.
    """

    info = _active_voice_info()
    if info is not None and not info.file_backed:
        return [_check_checkpoint("TTS checkpoint", LOCAL_TTS_MODEL_PATH, TTS_MODEL)]
    return [
        _check_checkpoint("clone checkpoint", LOCAL_VOICE_CLONE_MODEL_PATH, VOICE_CLONE_MODEL),
        _check_checkpoint("design checkpoint", LOCAL_VOICE_DESIGN_MODEL_PATH, VOICE_DESIGN_MODEL),
    ]


def _check_active_voice() -> CheckResult:
    """Confirm ACTIVE_VOICE points at something, without decoding it.

    The backend's catalog answers for everything it lists; the fallbacks keep
    the two path forms working — an audio file anywhere on disk, or a bare
    filename inside the voices directory — exactly as ``resolve_voice``
    accepts them.
    """

    info = _active_voice_info()
    if info is not None:
        return CheckResult("narrator voice", OK, f"{info.kind}: {info.spec}")

    spec = str(ACTIVE_VOICE)
    candidate = Path(spec)
    if candidate.is_file() and candidate.suffix.lower() in AUDIO_SUFFIXES:
        return CheckResult("narrator voice", OK, f"recording {spec}")
    if (VOICES_DIR / candidate.name).is_file():
        return CheckResult("narrator voice", OK, f"recording {VOICES_DIR / candidate.name}")
    return CheckResult(
        "narrator voice",
        FAIL,
        f"ACTIVE_VOICE = {spec!r} names nothing in the voice library and is "
        "not an audio file. Design one in the UI or with design_voice.py, or "
        "pick any voice from the picker.",
    )


def _check_preparation_provider(
    provider: str = DEFAULT_PREPARATION_PROVIDER,
    model: str = DEFAULT_PREPARATION_MODEL,
    base_url: str = DEFAULT_PROVIDER_BASE_URL,
) -> CheckResult:
    """Ask the configured LLM provider whether it can actually serve requests.

    For Ollama this confirms both that the server answers and that the model
    is pulled — the two ways a prepare run dies after extraction already ran.
    A model that is merely missing is fetched here rather than reported: this
    is the earliest point at which the download can happen, and it is the one
    unmet dependency the project can satisfy without the user.
    """

    from .preparation import create_provider, provider_descriptor
    from .preparation.providers.base import (
        ProviderResponseError,
        ProviderUnavailableError,
    )

    name = f"preparation LLM ({provider}: {model})"
    try:
        descriptor = provider_descriptor(provider)
    except ValueError as exc:
        return CheckResult(name, FAIL, str(exc))
    if model not in descriptor.models:
        return CheckResult(
            name,
            WARN,
            f"{model!r} is not in PREPARATION_PROVIDERS[{provider!r}]['models']; "
            "the frontend cannot offer it.",
        )
    # A hosted provider fails at the first LLM call, long after extraction has
    # run; a missing key is knowable now.
    missing = descriptor.missing_requirement()
    if missing:
        return CheckResult(name, FAIL, missing)
    pulled: list[str] = []

    def on_pull_progress(message: str) -> None:
        pulled.append(message)
        print(message, flush=True)

    settings = {"model": model, "base_url": base_url, "timeout": 15.0}
    try:
        # Only a local adapter installs models; a hosted one has no such knob,
        # so the callback is offered rather than required.
        try:
            instance = create_provider(provider, **settings, on_pull_progress=on_pull_progress)
        except TypeError:
            instance = create_provider(provider, **settings)
    except ValueError as exc:
        return CheckResult(name, FAIL, str(exc))
    try:
        instance.check_available()
    except (ProviderUnavailableError, ProviderResponseError) as exc:
        return CheckResult(name, FAIL, str(exc))
    finally:
        instance.close()
    if pulled:
        return CheckResult(name, WARN, f"{base_url} — pulled {model} on this run")
    return CheckResult(name, OK, base_url)


# What transcription actually loads from the Whisper repository.  The repo
# ships the same weights several times over — fp16 and fp32 safetensors, .bin,
# flax — and transformers reads only the consolidated fp16 ``model.safetensors``
# plus the small config and tokenizer files, so these patterns fetch ~3 GB
# instead of the ~25 GB full snapshot.
_ASR_FILE_PATTERNS = ["*.json", "*.txt", "model.safetensors"]


def _check_asr_cache() -> CheckResult:
    """Whisper downloads on first use; fetch it now instead of stalling a run.

    The download goes to the Hugging Face cache — where ``from_pretrained``
    will look for it — not to a local directory, and huggingface_hub shows a
    tqdm bar per file.  A failed download is a warning, not an error: the
    first imported recording still fetches lazily as before.
    """

    if not REFERENCE_TRANSCRIBE:
        return CheckResult("ASR model", OK, "disabled (REFERENCE_TRANSCRIBE = False)")
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(ASR_MODEL, "config.json")
    except Exception:
        return CheckResult("ASR model", WARN, f"Could not inspect the cache for {ASR_MODEL}.")
    if isinstance(cached, str):
        return CheckResult("ASR model", OK, f"{ASR_MODEL} (cached)")
    print(f"  ASR model: downloading {ASR_MODEL} to the Hugging Face cache", flush=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(ASR_MODEL, allow_patterns=_ASR_FILE_PATTERNS)
    except Exception as exc:  # noqa: BLE001 - any failure means "not downloaded"
        return CheckResult(
            "ASR model",
            WARN,
            f"{ASR_MODEL} is not cached and downloading it failed ({exc}); "
            "the first imported recording will retry.",
        )
    return CheckResult("ASR model", OK, f"{ASR_MODEL} — downloaded on this run")


def _check_output_dir(output_dir: Path = DEFAULT_OUTPUT_DIR) -> CheckResult:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".preflight-write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult("output directory", FAIL, f"Cannot write {output_dir}: {exc}")
    return CheckResult("output directory", OK, str(output_dir))


def run_preflight() -> list[CheckResult]:
    """Run every check and return the results, failures included."""

    results = [
        _check_ffmpeg(),
        _check_packages(),
        _check_cuda(),
        *_checkpoint_checks(),
        _check_active_voice(),
        _check_preparation_provider(),
        _check_asr_cache(),
        _check_output_dir(),
    ]
    return results


def format_report(results: list[CheckResult]) -> str:
    """Render results as an aligned terminal report."""

    marks = {OK: "✓", WARN: "!", FAIL: "✗"}
    width = max(len(result.name) for result in results)
    lines = [
        f" {marks[result.status]} {result.name.ljust(width)}  {result.detail}" for result in results
    ]
    failures = sum(result.status == FAIL for result in results)
    warnings = sum(result.status == WARN for result in results)
    if failures:
        lines.append(f"\n{failures} check(s) failed.")
    elif warnings:
        lines.append(f"\nReady, with {warnings} warning(s).")
    else:
        lines.append("\nAll checks passed.")
    return "\n".join(lines)


def passed(results: list[CheckResult]) -> bool:
    """Whether startup should proceed: warnings allowed, failures not."""

    return all(result.status != FAIL for result in results)


__all__ = ["CheckResult", "FAIL", "OK", "WARN", "format_report", "passed", "run_preflight"]
