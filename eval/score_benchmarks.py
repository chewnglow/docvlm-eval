#!/usr/bin/env python3
"""Score saved predictions for MMLongBench-Doc, LongDocURL, and SlideVQA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from benchmark_common import (
    extract_answer_from_text,
    group_mean,
    parse_literal,
    parse_pages,
    parse_predicted_pages,
    read_jsonl,
    score_longdocurl,
    score_mmlongbench,
    slidevqa_answer_scores,
    slidevqa_evidence_scores,
    summarize_metric,
    write_jsonl,
)


EXTRACTION_PROMPT = """Given the question and analysis, extract a concise answer.

Return exactly:
Extracted answer: [answer]
Answer format: [Integer|Float|String|List|None]

If the analysis says the question cannot be answered from the documents, use "Not answerable".
If the analysis only says it cannot read or understand the images/documents, use "Fail to answer".
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmlongbench-doc", "longdocurl", "slidevqa"], required=True)
    parser.add_argument("--predictions", required=True, help="Input JSONL from run_benchmarks.py or equivalent.")
    parser.add_argument("--output", required=True, help="Scored JSONL path.")
    parser.add_argument("--summary", required=True, help="Metric summary JSON path.")
    parser.add_argument("--prediction-field", default=None)
    parser.add_argument("--extract-model", default=None, help="Optional OpenAI-compatible answer extractor model.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--queue-dir",
        default=None,
        help="Optional distributed queue used to validate coverage and identify terminal failures.",
    )
    parser.add_argument(
        "--failed-output",
        default=None,
        help="Write unresolved records from terminally failed queue tasks to this JSONL file.",
    )
    return parser.parse_args()


def distributed_coverage(
    queue_dir: Path,
    prediction_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a drained queue and describe records excluded after terminal failure."""
    manifest = json.loads((queue_dir / "manifest.json").read_text())
    expected_count = int(manifest.get("record_count", 0))
    if expected_count == 0:
        raise SystemExit(
            f"Refusing to score an empty queue at {queue_dir}. Check the dataset metadata files and rebuild the queue."
        )

    records = read_jsonl(queue_dir / "records.jsonl")
    records_by_id = {str(record["id"]): record for record in records}
    if len(records_by_id) != expected_count:
        raise SystemExit(
            f"Queue manifest expects {expected_count} records, but {queue_dir / 'records.jsonl'} "
            f"contains {len(records_by_id)} unique records."
        )

    predicted_ids = {str(row["id"]) for row in prediction_rows}
    expected_ids = set(records_by_id)
    extra_ids = predicted_ids - expected_ids
    if extra_ids:
        preview = ", ".join(sorted(extra_ids)[:5])
        raise SystemExit(f"Predictions contain {len(extra_ids)} IDs outside this queue: {preview}")

    failure_by_record: dict[str, dict[str, Any]] = {}
    failed_tasks = sorted((queue_dir / "failed").glob("task-*.json"))
    for path in failed_tasks:
        task = json.loads(path.read_text())
        failure = {
            "task_id": str(task.get("task_id", path.stem)),
            "attempts": int(task.get("attempts", 0)),
            "last_error": str(task.get("last_error", "")),
        }
        for record_id in task.get("record_ids", []):
            failure_by_record[str(record_id)] = failure

    missing_ids = expected_ids - predicted_ids
    excluded_ids = missing_ids & set(failure_by_record)
    unexpected_missing = missing_ids - excluded_ids
    if unexpected_missing:
        preview = ", ".join(sorted(unexpected_missing)[:5])
        raise SystemExit(
            f"Refusing to score: {len(unexpected_missing)} records have neither predictions nor terminal "
            f"failed-task entries: {preview}"
        )

    failed_records: list[dict[str, Any]] = []
    for record_id in sorted(excluded_ids):
        record = dict(records_by_id[record_id])
        record["queue_failure"] = failure_by_record[record_id]
        failed_records.append(record)

    scored_count = len(predicted_ids)
    excluded_count = len(excluded_ids)
    return (
        {
            "status": "partial" if excluded_count else "complete",
            "expected_count": expected_count,
            "scored_count": scored_count,
            "excluded_failed_count": excluded_count,
            "failed_task_count": len(failed_tasks),
            "score_coverage": scored_count / expected_count,
        },
        failed_records,
    )


def extract_with_openai(args: argparse.Namespace, row: dict[str, Any]) -> str:
    if not args.extract_model:
        return extract_answer_from_text(row.get("response", row.get("prediction", "")))
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'openai'. Install eval/requirements-benchmark.txt first."
        ) from exc
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key for --extract-model.")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.extract_model,
        messages=[
            {"role": "user", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": f"Question: {row.get('question', '')}\nAnalysis: {row.get('response', '')}",
            },
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return extract_answer_from_text(response.choices[0].message.content or "")


def get_prediction(args: argparse.Namespace, row: dict[str, Any]) -> str:
    if args.prediction_field:
        return extract_answer_from_text(row.get(args.prediction_field, ""))
    for key in ("pred", "prediction", "extracted_answer", "final_answer"):
        if key in row:
            return extract_answer_from_text(row[key])
    return extract_with_openai(args, row)


def score_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        pred = get_prediction(args, row)
        item["pred"] = pred
        if args.dataset == "mmlongbench-doc":
            item["score"] = score_mmlongbench(row.get("answer"), pred, row.get("answer_format", "String"))
        elif args.dataset == "longdocurl":
            item["score"] = score_longdocurl(row.get("answer"), pred, row.get("answer_format", "String"))
        else:
            ans = slidevqa_answer_scores(row.get("answer"), pred)
            ev = slidevqa_evidence_scores(row.get("evidence_pages"), parse_predicted_pages(row))
            item.update(ans)
            item.update(ev)
            item["main_f1"] = ans["answer_f1"] * ev["evidence_f1"]
            item["main_em"] = ans["answer_em"] * ev["evidence_em"]
        scored.append(item)
    return scored


def mmlong_f1(rows: list[dict[str, Any]]) -> float:
    positives = [r for r in rows if r.get("answer") != "Not answerable"]
    predicted = [r for r in rows if r.get("pred") != "Not answerable"]
    if not positives or not predicted:
        return 0.0
    correct_positive = sum(float(r.get("score", 0.0)) for r in positives)
    recall = correct_positive / len(positives)
    precision = correct_positive / len(predicted)
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def summarize(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.dataset == "slidevqa":
        return {
            "dataset": args.dataset,
            "count": len(rows),
            "answer_f1": summarize_metric(rows, "answer_f1")["mean"],
            "answer_em": summarize_metric(rows, "answer_em")["mean"],
            "evidence_f1": summarize_metric(rows, "evidence_f1")["mean"],
            "evidence_em": summarize_metric(rows, "evidence_em")["mean"],
            "main_f1": summarize_metric(rows, "main_f1")["mean"],
            "main_em": summarize_metric(rows, "main_em")["mean"],
        }
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "count": len(rows),
        "accuracy": summarize_metric(rows, "score")["mean"],
        "by_answer_format": group_mean(rows, "answer_format"),
    }
    if args.dataset == "mmlongbench-doc":
        for row in rows:
            row["evidence_pages"] = parse_pages(row.get("evidence_pages"))
            row["evidence_sources"] = parse_literal(row.get("evidence_sources"), default=[])
        summary["f1"] = mmlong_f1(rows)
        summary["single_page"] = summarize_metric([r for r in rows if len(r.get("evidence_pages", [])) == 1])
        summary["cross_page"] = summarize_metric(
            [r for r in rows if len(r.get("evidence_pages", [])) != 1 and r.get("answer") != "Not answerable"]
        )
        summary["unanswerable"] = summarize_metric([r for r in rows if r.get("answer") == "Not answerable"])
        summary["by_evidence_source"] = group_mean(rows, "evidence_sources")
        summary["by_doc_type"] = group_mean(rows, "doc_type")
    else:
        for row in rows:
            row["evidence_pages"] = parse_pages(row.get("evidence_pages"))
            row["evidence_sources"] = parse_literal(row.get("evidence_sources"), default=[])
            row["subTask"] = parse_literal(row.get("subTask"), default=row.get("subTask", []))
            row["page_group"] = "single_page" if len(row.get("evidence_pages", [])) == 1 else "multi_page"
            row["element_group"] = "cross_element" if len(set(row.get("evidence_sources", []))) > 1 else "single_element"
        summary["by_task_tag"] = group_mean(rows, "task_tag")
        summary["by_question_type"] = group_mean(rows, "question_type")
        summary["by_evidence_source"] = group_mean(rows, "evidence_sources")
        summary["by_page_group"] = group_mean(rows, "page_group")
        summary["by_element_group"] = group_mean(rows, "element_group")
        summary["by_subtask"] = group_mean(rows, "subTask")
    return summary


def main() -> None:
    args = parse_args()
    if bool(args.queue_dir) != bool(args.failed_output):
        raise SystemExit("--queue-dir and --failed-output must be provided together.")
    rows = read_jsonl(Path(args.predictions))
    coverage = None
    if args.queue_dir:
        coverage, failed_records = distributed_coverage(Path(args.queue_dir), rows)
        write_jsonl(Path(args.failed_output), failed_records)
    scored = score_rows(args, rows)
    write_jsonl(Path(args.output), scored)
    summary = summarize(args, scored)
    if coverage is not None:
        summary["coverage"] = coverage
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
