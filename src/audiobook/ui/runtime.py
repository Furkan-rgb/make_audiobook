"""Shared plumbing for the frontend: one GPU slot, and readable progress.

The pipeline was written for a terminal.  Both helpers here bridge that to a
browser without changing it: :func:`load_model` keeps a single checkpoint
resident because three 1.7B models do not fit alongside each other, and
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

_loaded: tuple[str, Any] | None = None
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


def load_model(path: str) -> Any:
    """Return the model at *path*, evicting whichever one is resident.

    Design, clone and narration each want a different checkpoint, and switching
    tabs should not accumulate them in VRAM.  Re-requesting the model already
    loaded is free, which is what makes auditioning several passages quick.
    """

    global _loaded
    import torch
    from qwen_tts import Qwen3TTSModel

    if _loaded is not None and _loaded[0] == path:
        return _loaded[1]

    if _loaded is not None:
        print(f"Unloading {_loaded[0]}...")
        _loaded = None
        torch.cuda.empty_cache()

    print(f"Loading {path} on {torch.cuda.get_device_name(0)}...")
    model = Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16)
    _loaded = (path, model)
    return model


def unload_model() -> str:
    """Drop the resident checkpoint and return a note about what happened."""

    global _loaded
    if _loaded is None:
        return "No model is loaded."

    import torch

    name = _loaded[0]
    _loaded = None
    torch.cuda.empty_cache()
    return f"Unloaded {name}."


def loaded_model_name() -> str | None:
    """Name of the resident checkpoint, or ``None`` when the GPU is idle."""

    return _loaded[0] if _loaded is not None else None


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
    "load_model",
    "loaded_model_name",
    "stream_output",
    "unload_model",
]
