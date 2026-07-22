"""Shared plumbing for the frontend: one GPU slot, and readable progress.

The pipeline was written for a terminal.  The helpers here bridge that to a
browser without changing it: :func:`synthesis_provider` shares one TTS backend
instance so a single checkpoint stays resident (three 1.7B models do not fit
alongside each other, and the provider evicts on switch), and
:func:`stream_output` turns the pipeline's ``print``/``tqdm`` chatter into a log
the page can show while work is still running.
"""

from __future__ import annotations

import io
import queue
import sys
import threading
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Iterator

from ..config import DEFAULT_SYNTHESIS_PROVIDER
from ..synthesis.providers import SynthesisProvider, create_synthesis_provider

_provider: SynthesisProvider | None = None
_gpu_lock = threading.Lock()


def gpu_lock() -> threading.Lock:
    """The lock every GPU action holds, so runs queue instead of colliding.

    Gradio will happily run two handlers at once; two Qwen models loading at
    once is an out-of-memory error rather than a slow page.
    """

    return _gpu_lock


@contextmanager
def gpu_slot():
    """Hold the GPU for the duration of a block, saying so if there is a wait.

    Acquiring the lock silently is what makes a queued run look like a frozen
    page: nothing prints, so nothing reaches the log the browser is watching.
    Announcing the wait from inside the worker thread puts it in that log
    through the same :func:`stream_output` path as the run's own output.
    """

    lock = gpu_lock()
    if not lock.acquire(blocking=False):
        print("Waiting for the GPU — another run is using it...")
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


def synthesis_provider() -> SynthesisProvider:
    """The frontend's one long-lived TTS backend instance.

    Residency lives inside the provider — it keeps exactly one checkpoint in
    VRAM and evicts on switch — so sharing the instance across every tab is
    what makes repeated auditions quick and tab switches safe.  Design, clone
    and narration each want a different checkpoint, and switching tabs must
    not accumulate them.
    """

    global _provider
    if _provider is None:
        _provider = create_synthesis_provider(DEFAULT_SYNTHESIS_PROVIDER)
    return _provider


def unload_model() -> str:
    """Drop the resident checkpoint and return a note about what happened."""

    name = loaded_model_name()
    if name is None:
        return "No model is loaded."
    synthesis_provider().close()
    return f"Unloaded {name}."


def loaded_model_name() -> str | None:
    """Name of the resident checkpoint, or ``None`` when the GPU is idle."""

    if _provider is None:
        return None
    resident = getattr(_provider, "resident_checkpoint", None)
    return resident() if callable(resident) else None


class _Tee(io.TextIOBase):
    """Fan writes out to a queue for the browser and on to the terminal."""

    def __init__(self, sink: queue.Queue, original: Any) -> None:
        self._sink = sink
        self._original = original

    def write(self, text: str) -> int:
        if text:
            self._sink.put(text)
            self._original.write(text)
        return len(text)

    def flush(self) -> None:
        self._original.flush()


def stream_output(work: Callable[[], Any]) -> Iterator[tuple[str, Any]]:
    """Run *work* in a thread, yielding ``(log_so_far, result)`` as it goes.

    ``result`` is ``None`` until the call finishes, so a handler can bind the
    log to a textbox and the return value to whatever displays it.  Exceptions
    are formatted into the log rather than raised: a traceback in the page is
    more use than a Gradio error toast that hides which stage failed.

    Progress bars are carried by ``\\r`` rather than newlines, so each carriage
    return rewinds to the start of the last line the way a terminal would.
    """

    sink: queue.Queue = queue.Queue()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            with (
                redirect_stdout(_Tee(sink, sys.stdout)),
                redirect_stderr(_Tee(sink, sys.stderr)),
            ):
                box["result"] = work()
        except BaseException:  # surfaced in the log, not swallowed
            sink.put("\n" + traceback.format_exc())
        finally:
            sink.put(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    lines: list[str] = [""]
    while True:
        chunk = sink.get()
        if chunk is None:
            break
        for piece in chunk.splitlines(keepends=True):
            if piece.startswith("\r"):
                lines[-1] = piece.lstrip("\r").rstrip("\n")
            elif piece.endswith("\n"):
                lines[-1] += piece[:-1]
                lines.append("")
            else:
                lines[-1] += piece
        yield "\n".join(lines[-400:]), None

    thread.join()
    yield "\n".join(lines[-400:]), box.get("result")


__all__ = [
    "gpu_lock",
    "gpu_slot",
    "loaded_model_name",
    "stream_output",
    "synthesis_provider",
    "unload_model",
]
