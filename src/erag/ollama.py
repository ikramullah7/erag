"""Local Ollama chat adapter using the standard library; never pulls a model."""

import json
import time
from dataclasses import replace
from urllib import error, request

from .answer_config import AnswerConfig
from .answering import Generation


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaTimeout(RuntimeError):
    """A timed-out request; distinguish it from configuration or protocol errors."""


class OllamaTruncated(RuntimeError):
    """The generation exhausted its token budget without a completed answer."""


class OllamaGenerator:
    def __init__(self, cfg: AnswerConfig):
        self.cfg = cfg
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = request.Request(
            self.cfg.base_url.rstrip("/") + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(req, timeout=self.cfg.timeout_seconds) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise RuntimeError("Ollama response exceeds the response budget")
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError("expected JSON object")  # noqa: TRY004
            return value
        except error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama HTTP {exc.code}; check the installed model and local runtime"
            ) from exc
        except TimeoutError as exc:
            raise OllamaTimeout(
                "Local Ollama request timed out; reduce inference work or increase timeout_seconds"
            ) from exc
        except (error.URLError, OSError) as exc:
            raise RuntimeError("Cannot reach local Ollama; check that Ollama is running") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc

    def verify_model(self) -> str:
        installed = self._request("/api/tags").get("models", [])
        if not isinstance(installed, list) or any(not isinstance(m, dict) for m in installed):
            raise RuntimeError("Ollama returned an invalid model list")
        model = next((m for m in installed if m.get("name") == self.cfg.model), None)
        if model is None:
            raise RuntimeError(
                f"Configured local model is not installed: {self.cfg.model}; no automatic download attempted"
            )
        if model.get("digest") != self.cfg.model_digest:
            raise RuntimeError(
                "Installed model digest differs from the pinned answer configuration"
            )
        return f"ollama:{self.cfg.model}@{self.cfg.model_digest}"

    def review(self, messages, schema):
        review_cfg = replace(
            self.cfg,
            think=self.cfg.coverage_think,
            num_predict=self.cfg.num_predict
            if self.cfg.coverage_think
            else min(self.cfg.num_predict, 1024),
        )
        return OllamaGenerator(review_cfg).generate(messages, schema)

    def generate(self, messages: list[dict], schema: dict) -> Generation:
        started = time.perf_counter()
        try:
            return self._generate_once(messages, schema)
        except (OllamaTimeout, OllamaTruncated) as exc:
            if not self.cfg.reasoning_fallback or not self.cfg.think:
                raise
            initial_ms = (time.perf_counter() - started) * 1000
            recovered = self._generate_once(
                messages, schema, think=False, num_predict=min(self.cfg.num_predict, 512)
            )
            return Generation(
                recovered.payload,
                recovered.model_identity,
                {
                    **recovered.usage,
                    "reasoning_recovery": {
                        "initial_error_type": type(exc).__name__,
                        "initial_error": str(exc),
                        "initial_elapsed_ms": initial_ms,
                        "initial_num_predict": self.cfg.num_predict,
                        "fallback_num_predict": min(self.cfg.num_predict, 512),
                        "requests": 2,
                    },
                },
                (time.perf_counter() - started) * 1000,
            )

    def _generate_once(
        self, messages: list[dict], schema: dict, *, think=None, num_predict=None
    ) -> Generation:
        identity = self.verify_model()
        runtime = self._request("/api/version").get("version")
        started = time.perf_counter()
        response = self._request(
            "/api/chat",
            {
                "model": self.cfg.model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "think": self.cfg.think if think is None else think,
                "keep_alive": "5m",
                "options": {
                    "temperature": self.cfg.temperature,
                    "seed": self.cfg.seed,
                    "num_ctx": self.cfg.num_ctx,
                    "num_predict": self.cfg.num_predict if num_predict is None else num_predict,
                },
            },
        )
        if response.get("done") is True and response.get("done_reason") == "length":
            raise OllamaTruncated(
                "Ollama exhausted its token budget; truncated answers are rejected"
            )
        if response.get("done") is not True or response.get("done_reason") != "stop":
            raise RuntimeError("Ollama did not finish normally; truncated answers are rejected")
        message = response.get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("Ollama returned an invalid message")  # noqa: TRY004
        if message.get("tool_calls"):
            raise RuntimeError("Answer generation must not request tool execution")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response is missing answer content")  # noqa: TRY004
        try:
            payload = json.loads(content)
        except ValueError as exc:
            raise RuntimeError("Model did not return the required JSON answer") from exc
        # Detect a model tag changed while generation was in flight.
        self.verify_model()
        usage = {
            k: response.get(k)
            for k in (
                "prompt_eval_count",
                "eval_count",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "total_duration",
            )
        }
        usage["ollama_version"] = runtime
        return Generation(payload, identity, usage, (time.perf_counter() - started) * 1000)
