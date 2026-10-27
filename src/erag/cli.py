"""CLI for indexing, inspecting retrieval, validation and explicit-split evaluation."""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from .answer_config import load_answer_config
from .answer_evaluation import evaluate_answers, format_answer_checks
from .answer_metrics import abstention_only, calibrate_judge, evaluate_semantics, format_metrics
from .answering import answer_question
from .config import load_config
from .diagnostics import configure_logging, event
from .evaluation import evaluate, format_target_table, quality_gate
from .generators import create_generator
from .models import CrossEncoderReranker, SentenceEmbedder
from .ollama import OllamaGenerator
from .pipeline import Engine, atomic_json, build_index
from .storage import QdrantStore


def main() -> int:
    parser = argparse.ArgumentParser(prog="erag")
    parser.add_argument("--config", default=None)
    parser.add_argument("--qdrant-url")
    sub = parser.add_subparsers(dest="command", required=True)
    semantics = sub.add_parser(
        "score-answers", help="Score saved answers with a calibrated local semantic judge"
    )
    semantics.add_argument("--answers", required=True)
    semantics.add_argument(
        "--references", default="data/stage-1.1/answer-semantic-references-v1.json"
    )
    semantics.add_argument("--judge-config", default="config/answer-judge.yaml")
    semantics.add_argument("--calibration", default="data/stage-1.1/semantic-calibration-v1.json")
    semantics.add_argument("--output", required=True)
    sub.add_parser("index")
    sub.add_parser("validate")
    delete = sub.add_parser("delete")
    delete.add_argument("document_id")
    sub.add_parser("restore")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    search.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    search.add_argument("--top-k", type=int, default=10)
    answer = sub.add_parser(
        "answer", help="Answer from the configured document stage with source citations"
    )
    answer.add_argument("question")
    answer.add_argument("--answer-config", default="config/answer-real-stage1-codex.yaml")
    answer.add_argument(
        "--detail",
        action="store_true",
        help="Show full selected evidence instead of a brief answer",
    )
    answer.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    answer.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    answer.add_argument(
        "--json", action="store_true", help="include claim evidence and diagnostics as JSON"
    )
    answers_eval = sub.add_parser("evaluate-answers", help="Check generated answers and abstention")
    answers_eval.add_argument("--answer-config", default="config/answer.yaml")
    answers_eval.add_argument("--cases", default=None)
    answers_eval.add_argument("--output", required=True)
    answers_eval.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    answers_eval.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    ev = sub.add_parser("evaluate")
    ev.add_argument(
        "--split",
        choices=[
            "tuning",
            "heldout",
            "tuning-v2",
            "heldout-v2",
            "tuning-v3",
            "heldout-v3",
            "tuning-v4",
            "heldout-v4",
        ],
        required=True,
    )
    ev.add_argument("--experiment-id", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    ev.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    ev.add_argument(
        "--check-targets",
        action="store_true",
        help="print the five-metric table; exit 2 if selected strategy fails quality gates",
    )
    args = parser.parse_args()
    configure_logging()
    store = None
    try:
        if args.command == "score-answers":
            output = Path(args.output)
            calibration_output = output.with_name(output.stem + "-calibration.json")
            if output.exists() or output.with_suffix(".partial.json").exists():
                raise ValueError("preserve earlier semantic evaluations; choose a new output")
            judge_cfg = load_answer_config(args.judge_config)
            judge = OllamaGenerator(judge_cfg)
            calibration = calibrate_judge(judge, judge_cfg, args.calibration, calibration_output)
            if not calibration["passed"]:
                blocked = abstention_only(args.answers, args.references)
                blocked["calibration"] = calibration
                atomic_json(output, blocked)
                print(format_metrics(blocked))
                print(
                    f"Judge calibration failed; semantic scores not produced. See {calibration_output}"
                )
                return 2
            report = evaluate_semantics(args.answers, args.references, judge, judge_cfg, output)
            report["calibration"] = calibration
            atomic_json(output, report)
            print(format_metrics(report))
            return 0 if report["summary"]["semantic_scoring_complete"] else 2
        answer_cfg = (
            load_answer_config(args.answer_config)
            if args.command in {"answer", "evaluate-answers"}
            else None
        )
        if args.command == "answer" and args.detail:
            answer_cfg = replace(answer_cfg, brief_answers=False)
        config_path = args.config or (
            answer_cfg.retrieval_config if answer_cfg else "config/default.yaml"
        )
        cfg = load_config(config_path)
        rerank = cfg.rerank if getattr(args, "rerank", None) is None else args.rerank
        if args.qdrant_url:
            cfg = replace(cfg, qdrant_url=args.qdrant_url)
        store = QdrantStore(path=str(Path(cfg.state) / "qdrant"), url=cfg.qdrant_url)
        started = time.perf_counter()
        needs_embedder = (
            args.command in {"index", "evaluate", "delete", "restore"}
            or args.command in {"search", "answer", "evaluate-answers"}
            and (args.mode or cfg.mode) != "bm25"
        )
        embedder = SentenceEmbedder(cfg) if needs_embedder else None
        reranker = (
            CrossEncoderReranker(cfg)
            if args.command == "evaluate"
            or args.command in {"search", "answer", "evaluate-answers"}
            and rerank
            else None
        )
        event("providers_loaded", elapsed_ms=(time.perf_counter() - started) * 1000)
        if args.command in {"index", "delete", "restore"}:
            deleted = None
            if args.command == "delete":
                existing = Engine(cfg, store, embedder)
                deleted = existing.snapshot.get("deleted_documents", []) + [args.document_id]
            elif args.command == "restore":
                deleted = []
            snapshot = build_index(cfg, embedder, store, deleted)
            print(
                json.dumps(
                    {k: v for k, v in snapshot.items() if k not in {"chunks", "documents"}},
                    indent=2,
                )
            )
        else:
            engine = Engine(cfg, store, embedder, reranker)
            if args.command == "evaluate-answers":
                report = evaluate_answers(
                    engine,
                    create_generator(answer_cfg),
                    answer_cfg,
                    args.cases or str(Path(cfg.dataset) / "answer-evaluation-v1.json"),
                    args.output,
                    args.mode,
                    rerank,
                )
                print(format_answer_checks(report))
                return 0 if report["passed"] else 2
            elif args.command == "answer":
                result = answer_question(
                    engine,
                    create_generator(answer_cfg),
                    answer_cfg,
                    args.question,
                    args.mode,
                    rerank,
                )
                if args.json:
                    print(json.dumps(result, indent=2, allow_nan=False))
                elif result["status"] in {"provider_error", "invalid_response"}:
                    detail = (
                        result.get("error")
                        or result.get("validation_error")
                        or result.get("answer")
                    )
                    print(
                        f"Answer failed ({result['status']}): {detail or 'No valid response returned.'}",
                        file=sys.stderr,
                    )
                else:
                    print(result["answer"])
                    cited = {
                        e["source_id"] for claim in result["claims"] for e in claim["citations"]
                    }
                    if cited:
                        print("\nSources:")
                        for source in result["sources"]:
                            if source["source_id"] in cited:
                                print(
                                    f"[{source['source_id']}] {source['chunk']['source']} — {source['chunk']['heading']}"
                                )
                return 2 if result["status"] in {"invalid_response", "provider_error"} else 0
            elif args.command == "search":
                print(
                    json.dumps(
                        engine.search(args.query, args.mode, rerank, args.top_k),
                        indent=2,
                        allow_nan=False,
                    )
                )
            elif args.command == "validate":
                print(
                    json.dumps(
                        {
                            "valid": True,
                            "chunks": len(engine.chunks),
                            "corpus_hash": engine.snapshot["corpus_hash"],
                        }
                    )
                )
            else:
                output = Path(args.output)
                if output.exists():
                    raise ValueError("experiment output already exists; choose a new path")
                report = evaluate(
                    engine,
                    Path(cfg.dataset) / f"{args.split}.json",
                    output,
                    args.experiment_id,
                    args.split,
                )
                if args.check_targets:
                    strategy = (args.mode or cfg.mode) + ("+rerank" if rerank else "")
                    summary = report["summary"][strategy]
                    print(f"Stage {cfg.stage} | {args.split} | {strategy} | chunk level")
                    print(format_target_table(summary))
                    return 0 if quality_gate(summary, rerank)["passed"] else 2
                print(json.dumps(report["summary"], indent=2))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        event("error", error_type=type(exc).__name__, message=str(exc))
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    sys.exit(main())
