"""Ollama implementation of the narration-preparation provider protocol."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..prompting import RESPONSE_JSON_SCHEMA, build_messages
from ..types import (
    DEFAULT_PROMPT_VERSION,
    PreparationEdit,
    PreparationRequest,
    PreparationResult,
    ProviderMetadata,
)
from .base import ProviderResponseError, ProviderUnavailableError


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class OllamaProvider:
    """Prepare narration prose through a local Ollama ``/api/chat`` call."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 300.0,
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 8192,
        num_predict: int = 4096,
        keep_alive: str | int = "10m",
        unload_on_close: bool = True,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model cannot be blank")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be positive")
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

    def check_available(self) -> None:
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
        if not (self._model_aliases(self.model) & installed):
            raise ProviderUnavailableError(
                f"Ollama is running, but model {self.model!r} is not installed. "
                f"Run: ollama pull {self.model}"
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
            raise ProviderResponseError(
                "Ollama message.content did not satisfy the JSON output contract"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("prepared_text"), str
        ):
            raise ProviderResponseError("Structured response omitted prepared_text")
        edits_payload = payload.get("edits", [])
        warnings_payload = payload.get("warnings", [])
        if not isinstance(edits_payload, list) or not all(
            isinstance(item, dict) for item in edits_payload
        ):
            raise ProviderResponseError("Structured response edits must be an array")
        if not isinstance(warnings_payload, list) or not all(
            isinstance(item, str) for item in warnings_payload
        ):
            raise ProviderResponseError("Structured response warnings must be strings")
        return PreparationResult(
            prepared_text=payload["prepared_text"].strip(),
            edits=[PreparationEdit.from_dict(item) for item in edits_payload],
            warnings=warnings_payload,
            provider_metadata=self.metadata,
        )

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
