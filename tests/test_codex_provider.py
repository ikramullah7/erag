"""Provider switching and Codex subprocess boundaries, without live cloud inference."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from erag.answer_config import AnswerConfig
from erag.codex import CodexGenerator
from erag.generators import create_generator
from erag.ollama import OllamaGenerator


def config(**changes):
    return replace(AnswerConfig(model="qwen3:14b", model_digest="a" * 64), **changes)


def test_provider_flag_selects_codex_without_changing_local_model_settings():
    local = config()
    assert isinstance(create_generator(local), OllamaGenerator)
    remote = replace(local, provider="codex")
    assert isinstance(create_generator(remote), CodexGenerator)
    assert remote.model == local.model and remote.model_digest == local.model_digest


@pytest.mark.parametrize(
    "changes", [{"provider": "unknown"}, {"codex_model": "--unsafe"}, {"codex_binary": ""}]
)
def test_invalid_provider_settings_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


def stub_cli(monkeypatch, *, failure=None):
    monkeypatch.setattr("erag.codex.shutil.which", lambda name: "/fake codex")
    calls = []

    def run(args, **kwargs):
        if args[-1] == "--version":
            return SimpleNamespace(stdout="codex-cli test-version", returncode=0)
        calls.append((args, kwargs))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 1)
        if failure == "exit":
            return SimpleNamespace(returncode=1, stdout="", stderr="private diagnostic")
        output = Path(args[args.index("--output-last-message") + 1])
        output.write_text("not json" if failure == "json" else json.dumps({"evidence_ids": ["E1"]}))
        events = [{"type": "turn.completed", "usage": {"input_tokens": 123, "output_tokens": 8}}]
        if failure == "tool":
            events.insert(0, {"type": "item.started", "item": {"type": "command_execution"}})
        if failure == "incomplete":
            events = []
        return SimpleNamespace(
            returncode=0, stdout="\n".join(json.dumps(e) for e in events), stderr=""
        )

    monkeypatch.setattr("erag.codex.subprocess.run", run)
    return calls


def test_codex_uses_isolated_readonly_structured_stdin_and_records_identity(monkeypatch):
    calls = stub_cli(monkeypatch)
    provider = CodexGenerator(config(provider="codex", codex_model="configured-codex-model"))
    result = provider.generate(
        [{"role": "user", "content": "$(touch unsafe) source-only question"}], {"type": "object"}
    )
    args, kwargs = calls[0]
    assert args[0] == "/fake codex" and args[-1] == "-"
    assert "--ignore-user-config" in args and "--ephemeral" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "features.shell_tool=false" in args and 'web_search="disabled"' in args
    assert args[args.index("--model") + 1] == "configured-codex-model"
    assert "$(touch unsafe)" in kwargs["input"] and not any("$(touch" in a for a in args)
    assert kwargs.get("shell", False) is False
    assert not Path(kwargs["cwd"]).exists()
    assert result.payload == {"evidence_ids": ["E1"]}
    assert result.usage["input_tokens"] == 123
    assert result.model_identity == "codex-cli:configured-codex-model@codex-cli test-version"


@pytest.mark.parametrize("failure", ["timeout", "exit", "json", "tool", "incomplete"])
def test_codex_failure_never_exposes_answer_or_falls_back(monkeypatch, failure):
    stub_cli(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError) as exc:
        CodexGenerator(config(provider="codex")).generate([], {"type": "object"})
    assert "private diagnostic" not in str(exc.value)


def test_missing_codex_has_actionable_failure_without_model_download(monkeypatch):
    monkeypatch.setattr("erag.codex.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="not installed"):
        CodexGenerator(config(provider="codex")).generate([], {})
