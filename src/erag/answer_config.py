"""Answer-phase settings, separate from the unchanged retrieval configuration."""

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class AnswerConfig:
    model: str
    model_digest: str
    base_url: str = "http://127.0.0.1:11434"
    retrieval_config: str = "config/answer-retrieval.yaml"
    context_chunks: int = 6
    context_max_chars: int = 16000
    num_ctx: int = 8192
    num_predict: int = 1024
    timeout_seconds: float = 180
    temperature: float = 0
    seed: int = 0
    validation_retries: int = 0
    align_quote_fragments: bool = False

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty local model name")
        if not isinstance(self.model_digest, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.model_digest
        ):
            raise ValueError("model_digest must pin the full installed model SHA-256")
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be a local HTTP URL")  # noqa: TRY004
        url = urlparse(self.base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("this phase supports a loopback HTTP Ollama endpoint only")
        for key in ("context_chunks", "context_max_chars", "num_ctx", "num_predict", "seed"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if type(self.align_quote_fragments) is not bool:
            raise ValueError("align_quote_fragments must be boolean")
        if type(self.validation_retries) is not int or not 0 <= self.validation_retries <= 1:
            raise ValueError("validation_retries must be 0 or 1")
        if not 1 <= self.context_chunks <= 20 or not 100 <= self.context_max_chars <= 32000:
            raise ValueError("invalid answer context budget")
        if not 2048 <= self.num_ctx <= 32768 or not 64 <= self.num_predict <= 4096:
            raise ValueError("invalid model token budget")
        if self.num_predict >= self.num_ctx:
            raise ValueError("generation budget must be smaller than model context")
        for key in ("timeout_seconds", "temperature"):
            v = getattr(self, key)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f"{key} must be finite")
        if not 1 <= self.timeout_seconds <= 600 or not 0 <= self.temperature <= 1:
            raise ValueError("invalid timeout or temperature")
        if not isinstance(self.retrieval_config, str) or not self.retrieval_config.strip():
            raise ValueError("retrieval_config must be a nonempty path")


def load_answer_config(path: str | Path) -> AnswerConfig:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ValueError("invalid answer YAML configuration") from exc
    if not isinstance(data, dict):
        raise ValueError("answer configuration must be a mapping")  # noqa: TRY004
    unknown = set(data) - {f.name for f in fields(AnswerConfig)}
    if unknown:
        raise ValueError(f"unknown answer configuration fields: {sorted(unknown)}")
    if not {"model", "model_digest"} <= data.keys():
        raise ValueError("answer configuration requires model and model_digest")
    return AnswerConfig(**data)
