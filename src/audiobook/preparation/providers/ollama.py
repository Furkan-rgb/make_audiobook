"""Ollama implementation of the narration-preparation provider protocol."""

from __future__ import annotations

import json
import socket
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..prompting import (
    RESPONSE_JSON_SCHEMA,
    build_messages,
    parse_structured_response,
)
from ..types import (
    DEFAULT_PROMPT_VERSION,
    PreparationRequest,
    PreparationResult,
    ProviderMetadata,
)
from .base import ProviderDescriptor, ProviderResponseError, ProviderUnavailableError


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:12b"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
# A pull moves several gigabytes over the network; the ceiling exists to catch a
# wedged server, not to bound a slow download.
DEFAULT_PULL_TIMEOUT_SECONDS = 3600.0

PullProgress = Callable[[str], None]


def _default_pull_progress(message: str) -> None:
    print(message, flush=True)


def _format_size(num_bytes: int) -> str:
    """Bytes at a scale that stays readable for both a manifest and a 12 GB blob."""

    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.0f} KB"
    mb = kb / 1024
    if mb < 1000:
        return f"{mb:.0f} MB"
    return f"{mb / 1024:.1f} GB"


def _format_pull_event(model: str, event: dict[str, Any]) -> str:
    """One human-readable progress line, or "" for an event worth skipping.

    Progress is a percentage in steps of five against a fixed total, not a
    running byte count.  Caller-side deduplication is by line, and a counter
    that moves every event would defeat it: a single layer would emit hundreds
    of near-identical lines, readable while scrolling past but useless in a log.
    """

    status = event.get("status")
    if not isinstance(status, str) or not status:
        return ""
    total = event.get("total")
    completed = event.get("completed")
    if isinstance(total, int) and isinstance(completed, int) and total > 0:
        percent = min(100, int(completed * 100 / total)) // 5 * 5
        return f"  pulling {model}: {status} {percent}% of {_format_size(total)}"
    return f"  pulling {model}: {status}"


def _configured() -> dict[str, Any]:
    """This adapter's config entry, if the project defines one.

    Imported lazily so the preparation package keeps working standalone: the
    constants above are the fallback when there is no config to read.
    """

    try:
        from ...config import PREPARATION_PROVIDERS
    except ImportError:  # pragma: no cover - only when used outside the project
        return {}
    entry = PREPARATION_PROVIDERS.get("ollama")
    return dict(entry) if isinstance(entry, dict) else {}


class OllamaProvider:
    """Prepare narration prose through a local Ollama ``/api/chat`` call."""

    @classmethod
    def describe(cls) -> ProviderDescriptor:
        entry = _configured()
        models = tuple(entry.get("models") or (DEFAULT_OLLAMA_MODEL,))
        return ProviderDescriptor(
            name="ollama",
            label="Ollama (local)",
            models=models,
            default_model=models[0],
            base_url=str(entry.get("base_url") or DEFAULT_OLLAMA_BASE_URL),
            api_key_env=None,
            local=True,
        )

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 300.0,
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 8192,
        # A well-behaved response is a handful of short edits, far under this.
        # The ceiling is a runaway guard, not a target, and a model that runs
        # into it loses the whole unit to a truncated JSON string — so it stays
        # generous even though the expected output shrank.
        num_predict: int = 4096,
        keep_alive: str | int = "10m",
        unload_on_close: bool = True,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        auto_pull: bool | None = None,
        pull_timeout: float = DEFAULT_PULL_TIMEOUT_SECONDS,
        on_pull_progress: PullProgress | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model cannot be blank")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be positive")
        if pull_timeout <= 0:
            raise ValueError("Ollama pull timeout must be positive")
        if num_ctx <= 0 or num_predict <= 0:
            raise ValueError("Ollama context and output limits must be positive")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama base_url must be an http(s) URL")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.unload_on_close = unload_on_close
        self.prompt_version = prompt_version
        self.auto_pull = (
            auto_pull if auto_pull is not None else bool(_configured().get("auto_pull", True))
        )
        self.pull_timeout = pull_timeout
        self.on_pull_progress = on_pull_progress or _default_pull_progress
        self._closed = False
        self._used = False

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="ollama",
            model=self.model,
            prompt_version=self.prompt_version,
            base_url=self.base_url,
            parameters={
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "think": False,
                "structured_output": True,
            },
        )

    def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + endpoint,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace").strip()
            raise ProviderResponseError(
                f"Ollama returned HTTP {exc.code} for {endpoint}: "
                f"{detail or exc.reason}"
            ) from exc
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise ProviderUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Start it with "
                f"`ollama serve`, then pull the model with "
                f"`ollama pull {self.model}`. Details: {exc}"
            ) from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ProviderResponseError("Ollama response exceeded the 16 MiB limit")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError("Ollama returned malformed JSON") from exc
        if not isinstance(parsed, dict):
            raise ProviderResponseError("Ollama response must be a JSON object")
        if parsed.get("error"):
            raise ProviderResponseError(f"Ollama error: {parsed['error']}")
        return parsed

    @staticmethod
    def _model_aliases(name: str) -> set[str]:
        aliases = {name}
        if name.endswith(":latest"):
            aliases.add(name.removesuffix(":latest"))
        else:
            aliases.add(name + ":latest")
        return aliases

    def _model_installed(self) -> bool:
        response = self._request_json("GET", "/api/tags", timeout=min(self.timeout, 15.0))
        models = response.get("models")
        if not isinstance(models, list):
            raise ProviderResponseError("Ollama /api/tags omitted its models list")
        installed: set[str] = set()
        for item in models:
            if isinstance(item, dict):
                for key in ("name", "model"):
                    if isinstance(item.get(key), str):
                        installed.update(self._model_aliases(item[key]))
        return bool(self._model_aliases(self.model) & installed)

    def pull(self) -> None:
        """Fetch the model into the running server, reporting progress.

        ``/api/pull`` streams newline-delimited status objects rather than a
        single document, so this cannot go through :meth:`_request_json`: the
        point is to report progress while the bytes arrive, not after.
        """

        body = json.dumps({"model": self.model, "stream": True}).encode("utf-8")
        request = Request(
            self.base_url + "/api/pull",
            data=body,
            headers={"Accept": "application/x-ndjson", "Content-Type": "application/json"},
            method="POST",
        )
        last_line = ""
        try:
            with urlopen(request, timeout=self.pull_timeout) as response:
                for raw in response:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ProviderResponseError(
                            "Ollama /api/pull returned malformed JSON"
                        ) from exc
                    if not isinstance(event, dict):
                        raise ProviderResponseError(
                            "Ollama /api/pull events must be JSON objects"
                        )
                    if event.get("error"):
                        raise ProviderResponseError(
                            f"Ollama could not pull {self.model!r}: {event['error']}"
                        )
                    message = _format_pull_event(self.model, event)
                    if message and message != last_line:
                        self.on_pull_progress(message)
                        last_line = message
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace").strip()
            raise ProviderResponseError(
                f"Ollama returned HTTP {exc.code} pulling {self.model!r}: "
                f"{detail or exc.reason}"
            ) from exc
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise ProviderUnavailableError(
                f"The pull of {self.model!r} from Ollama at {self.base_url} did "
                f"not complete. Retry, or run `ollama pull {self.model}` "
                f"manually. Details: {exc}"
            ) from exc

    def check_available(self) -> None:
        """Ensure the server answers and the model is there, pulling if not.

        A missing model is the one dependency this project can resolve on its
        own — the server is up and ``/api/pull`` is a single call — so it is
        fetched rather than reported, unless ``auto_pull`` says otherwise.
        """

        if self._model_installed():
            return
        if not self.auto_pull:
            raise ProviderUnavailableError(
                f"Ollama is running, but model {self.model!r} is not installed. "
                f"Run: ollama pull {self.model}"
            )
        self.on_pull_progress(
            f"  {self.model} is not installed; pulling it from the Ollama library."
        )
        self.pull()
        if not self._model_installed():
            raise ProviderUnavailableError(
                f"Ollama reported the pull of {self.model!r} as finished, but the "
                "model is still not installed. Check the name against "
                "`ollama list` and the Ollama library."
            )

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        if self._closed:
            raise ProviderUnavailableError("This OllamaProvider has been closed")
        if request.prompt_version != self.prompt_version:
            raise ValueError(
                "Request prompt_version does not match the Ollama provider "
                f"({request.prompt_version!r} != {self.prompt_version!r})"
            )
        response = self._request_json(
            "POST",
            "/api/chat",
            {
                "model": self.model,
                "messages": build_messages(request),
                "stream": False,
                "think": False,
                "format": RESPONSE_JSON_SCHEMA,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            },
        )
        self._used = True
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderResponseError("Ollama response omitted message.content")
        content = message["content"].strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].strip() == "```":
                lines.pop()
            content = "\n".join(lines)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            # Truncation looks identical to malformed output once the JSON
            # parser has failed, and the two need opposite responses: raise the
            # ceiling, or distrust the model. Ollama says which one it was.
            if response.get("done_reason") == "length":
                raise ProviderResponseError(
                    f"Ollama stopped {self.model!r} at the {self.num_predict}-token "
                    "output limit, leaving the edit list unfinished. Raise "
                    "num_predict, or shorten the prose units so each one needs "
                    "fewer edits."
                ) from exc
            raise ProviderResponseError(
                "Ollama message.content did not satisfy the JSON output contract"
            ) from exc
        try:
            result = parse_structured_response(payload)
        except ValueError as exc:
            raise ProviderResponseError(str(exc)) from exc
        result.provider_metadata = self.metadata
        return result

    def unload(self) -> None:
        if self._closed or not self._used:
            return
        self._request_json(
            "POST",
            "/api/generate",
            {"model": self.model, "keep_alive": 0, "stream": False},
            timeout=min(self.timeout, 30.0),
        )
        self._used = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.unload_on_close:
                self.unload()
        except (ProviderResponseError, ProviderUnavailableError):
            # Releasing GPU memory is best-effort and must never mask the
            # preparation result or the original provider error.
            pass
        finally:
            self._closed = True

    def __enter__(self) -> "OllamaProvider":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
