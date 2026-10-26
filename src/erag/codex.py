"""Optional Codex CLI answer adapter using the user's existing CLI authentication.

Only selected evidence/messages go to stdin. Run in an isolated temporary directory,
ignore user config and workspace instructions, disable command execution and search,
and reject tool activity. No silent fallback to another provider.
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .answering import Generation


class CodexGenerator:
    def __init__(self, cfg):
        self.cfg = cfg

    def review(self, messages, schema):
        return self.generate(messages, schema)

    def generate(self, messages, schema):
        binary = shutil.which(self.cfg.codex_binary)
        if binary is None:
            raise RuntimeError("Codex CLI is not installed or codex_binary is unavailable")
        started = time.perf_counter()
        try:
            version = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
            with tempfile.TemporaryDirectory(prefix="erag-codex-") as directory:
                root = Path(directory)
                schema_file, answer_file = root / "schema.json", root / "answer.json"
                schema_file.write_text(json.dumps(schema))
                command = [
                    binary,
                    "exec",
                    "--ignore-user-config",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--cd",
                    str(root),
                    "--json",
                    "--color",
                    "never",
                    "-c",
                    'approval_policy="never"',
                    "-c",
                    "features.shell_tool=false",
                    "-c",
                    "features.apps=false",
                    "-c",
                    'web_search="disabled"',
                    "-c",
                    "project_doc_max_bytes=0",
                    "--output-schema",
                    str(schema_file),
                    "--output-last-message",
                    str(answer_file),
                ]
                if self.cfg.codex_model:
                    command.extend(["--model", self.cfg.codex_model])
                command.append("-")
                prompt = (
                    "Perform only structured answer generation from the supplied messages. "
                    "Do not inspect files, execute commands, browse, call tools or use outside knowledge. "
                    "Apply the supplied system messages as the answer policy. Return only the requested JSON.\n"
                    + json.dumps({"messages": messages}, ensure_ascii=False)
                )
                result = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_seconds,
                    check=False,
                    cwd=root,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        "Codex answer failed; check CLI sign-in, model access and configuration"
                    )
                if len(result.stdout.encode()) > 2_000_000:
                    raise RuntimeError("Codex event output exceeds the response budget")
                usage, completed = {}, False
                for line in result.stdout.splitlines():
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                        raise TypeError("invalid Codex event")
                    if event.get("type") in {"error", "turn.failed"}:
                        raise RuntimeError("Codex answer turn failed")
                    if event.get("type") == "turn.completed":
                        completed = True
                        usage = event.get("usage", {})
                        if not isinstance(usage, dict):
                            raise TypeError("invalid Codex usage")
                    if event.get("type", "").startswith("item.") and event.get("item", {}).get(
                        "type"
                    ) not in {
                        "agent_message",
                        "reasoning",
                        "plan",
                    }:
                        raise RuntimeError("Codex answer attempted tool activity; output rejected")
                if not completed or not answer_file.is_file():
                    raise RuntimeError("Codex did not return a completed structured answer")
                with answer_file.open("rb") as handle:
                    raw = handle.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise RuntimeError("Codex answer exceeds the response budget")
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")  # noqa: TRY004
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Codex answer timed out") from exc
        except (subprocess.CalledProcessError, OSError) as exc:
            raise RuntimeError("Cannot start Codex CLI; check installation and sign-in") from exc
        except (TypeError, ValueError, UnicodeError) as exc:
            raise RuntimeError("Codex returned invalid structured output") from exc
        return Generation(
            payload,
            f"codex-cli:{self.cfg.codex_model or 'default'}@{version}",
            {
                **usage,
                "provider": "codex",
                "cli_version": version,
                "requested_model": self.cfg.codex_model,
            },
            (time.perf_counter() - started) * 1000,
        )
