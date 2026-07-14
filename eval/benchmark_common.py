#!/usr/bin/env python3
"""Shared dataset loading, image handling, and scoring utilities."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PAGE_COLUMNS = [f"page_{i}" for i in range(1, 21)]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def require_package(import_name: str, install_name: str | None = None) -> Any:
    try:
        return __import__(import_name)
    except ImportError as exc:
        package = install_name or import_name
        raise SystemExit(
            f"Missing dependency '{package}'. Install benchmark deps with a Python "
            f"environment that has: eval/requirements-benchmark.txt"
        ) from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def natural_key(value: str | Path) -> list[Any]:
    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_literal(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return ast.literal_eval(text)
    except Exception:
        return default if default is not None else value


def normalize_answer_format(fmt: Any) -> str:
    text = str(fmt or "String").strip().lower()
    if text in {"int", "integer"}:
        return "Integer"
    if text in {"float", "double", "number"}:
        return "Float"
    if text in {"list", "array"}:
        return "List"
    if text in {"none", "not answerable", "unanswerable"}:
        return "None"
    return "String"


def clean_string(value: Any) -> str:
    text = str(value).lower().strip()
    for suffix in ("miles", "mile", "million"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"^['\"]|['\"]$", "", text).strip()
    text = text.strip().lstrip("$").strip().rstrip("%").strip()
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min(distances[i1], distances[i1 + 1], distances_[-1]))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth: Any, prediction: Any, threshold: float = 0.5) -> float:
    gt = str(groundtruth)
    pred = str(prediction)
    dist = levenshtein_distance(gt, pred)
    length = max(len(gt.upper()), len(pred.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    return 0.0 if anls <= threshold else anls


def is_exact_match_type(text: str) -> bool:
    if "https://" in text:
        return True
    if text.endswith(".py") or text.endswith("ipynb"):
        return True
    if text.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", text):
        return True
    if "a.m." in text or "p.m." in text:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        return True
    return False


def is_float_equal(reference: Any, prediction: Any, include_percentage: bool = False) -> bool:
    def precision(value: Any) -> int:
        text = str(value)
        return len(text.split(".")[-1]) if "." in text else 3

    try:
        ref = float(str(reference).strip().rstrip("%").strip())
        pred = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False
    candidates = [ref / 100, ref, ref * 100] if include_percentage else [ref]
    for item in candidates:
        if math.isclose(item, pred, rel_tol=0.01):
            return True
        digits = max(min(precision(pred), precision(item)), 2)
        if round(pred, digits) == round(item, digits):
            return True
    return False


def is_floatish(value: Any) -> bool:
    try:
        float(str(value).strip().rstrip("%").strip())
        return True
    except Exception:
        return False


def parse_list_answer(value: Any) -> list[Any]:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    if parsed is None:
        return []
    text = str(parsed).strip()
    if not text:
        return []
    if "\n" in text:
        parts = [re.sub(r"^\s*[-*0-9.)]+\s*", "", p).strip() for p in text.splitlines()]
        return [p for p in parts if p]
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    return [parsed]


def score_scalar(reference: Any, prediction: Any, answer_format: str) -> float:
    fmt = normalize_answer_format(answer_format)
    if fmt == "Integer":
        try:
            return float(int(reference) == int(float(str(prediction).replace(",", ""))))
        except Exception:
            return 0.0
    if fmt == "Float":
        return float(is_float_equal(clean_string(reference), clean_string(prediction), include_percentage=True))
    gt = clean_string(reference)
    pred = clean_string(prediction)
    if fmt == "None":
        return float(gt == pred)
    if is_exact_match_type(gt):
        return float(gt == pred)
    return anls_compute(gt, pred)


def score_mmlongbench(reference: Any, prediction: Any, answer_format: str) -> float:
    fmt = normalize_answer_format(answer_format)
    if fmt != "List":
        return score_scalar(reference, prediction, fmt)
    gt = parse_list_answer(reference)
    pred = parse_list_answer(prediction)
    if len(gt) != len(pred):
        return 0.0
    gt_clean = sorted(clean_string(x) for x in gt)
    pred_clean = sorted(clean_string(x) for x in pred)
    if not gt_clean:
        return float(not pred_clean)
    if is_floatish(gt_clean[0]) or is_exact_match_type(gt_clean[0]):
        return float("-".join(gt_clean) == "-".join(pred_clean))
    return min(anls_compute(g, p) for g, p in zip(gt_clean, pred_clean))


def score_longdocurl(reference: Any, prediction: Any, answer_format: str) -> float:
    fmt = normalize_answer_format(answer_format)
    if fmt != "List":
        return score_scalar(reference, prediction, fmt)
    refs = sorted(parse_list_answer(reference), key=lambda x: clean_string(x))
    preds = sorted(parse_list_answer(prediction), key=lambda x: clean_string(x))
    if not refs:
        return float(not preds)
    if not preds:
        return 0.0
    element_scores = []
    for ref in refs:
        element_fmt = "Float" if is_floatish(ref) else "String"
        element_scores.append(max(score_scalar(ref, pred, element_fmt) for pred in preds))
    penalty = min(1.0, len(refs) / len(preds)) ** 0.5
    return sum(element_scores) / len(refs) * penalty


def normalize_squad_answer(text: Any) -> str:
    value = str(text).lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def slidevqa_answer_scores(reference: Any, prediction: Any) -> dict[str, float]:
    gt = normalize_squad_answer(reference)
    pred = normalize_squad_answer(prediction)
    em = float(gt == pred)
    gt_tokens = gt.split()
    pred_tokens = pred.split()
    if not gt_tokens or not pred_tokens:
        f1 = float(gt_tokens == pred_tokens)
    else:
        common = Counter(gt_tokens) & Counter(pred_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            f1 = 0.0
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
    return {"answer_em": em, "answer_f1": f1}


def parse_pages(value: Any) -> list[int]:
    parsed = parse_literal(value, default=[])
    if isinstance(parsed, (int, float)):
        return [int(parsed)]
    if isinstance(parsed, str):
        return [int(x) for x in re.findall(r"\d+", parsed)]
    pages = []
    for item in parsed or []:
        try:
            pages.append(int(item))
        except Exception:
            continue
    return sorted(set(pages))


def parse_predicted_pages(record: dict[str, Any]) -> list[int]:
    for key in ("pred_evidence_pages", "evidence_prediction", "pred_pages"):
        if key in record:
            return parse_pages(record[key])
    text = str(record.get("response", ""))
    match = re.search(r"evidence\s*pages?\s*:\s*(\[[^\]]+\]|[0-9,\s]+)", text, re.I)
    return parse_pages(match.group(1)) if match else []


def slidevqa_evidence_scores(reference_pages: Any, predicted_pages: Any) -> dict[str, float]:
    gt = set(parse_pages(reference_pages))
    pred = set(parse_pages(predicted_pages))
    em = float(gt == pred)
    if not gt or not pred:
        f1 = float(gt == pred)
    else:
        overlap = len(gt & pred)
        if overlap == 0:
            f1 = 0.0
        else:
            precision = overlap / len(pred)
            recall = overlap / len(gt)
            f1 = 2 * precision * recall / (precision + recall)
    return {"evidence_em": em, "evidence_f1": f1}


def summarize_metric(rows: list[dict[str, Any]], key: str = "score") -> dict[str, Any]:
    values = [float(row[key]) for row in rows if key in row]
    return {"count": len(values), "mean": sum(values) / len(values) if values else 0.0}


def group_mean(rows: list[dict[str, Any]], field: str, score_key: str = "score") -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if isinstance(value, list):
            for item in value:
                groups[str(item)].append(row)
        else:
            groups[str(value)].append(row)
    return {name: summarize_metric(items, score_key) for name, items in sorted(groups.items())}


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )


def parquet_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    pa = require_package("pyarrow")
    parquet = __import__("pyarrow.parquet", fromlist=["read_table"])
    for path in sorted(paths, key=natural_key):
        table = parquet.read_table(path)
        for row in table.to_pylist():
            yield row


def load_mmlongbench(dataset_root: Path, split: str = "train") -> Iterable[dict[str, Any]]:
    data_dir = dataset_root / "mmlongbench-doc" / "data"
    for row_index, row in enumerate(parquet_rows(data_dir.glob(f"{split}-*.parquet"))):
        doc_id = str(row["doc_id"])
        question = str(row["question"])
        row_id = row.get("question_id") or row.get("qa_id") or row.get("id")
        if row_id is None:
            digest = hashlib.sha1(f"{doc_id}\0{question}".encode("utf-8")).hexdigest()[:16]
            row_id = f"{Path(doc_id).stem}:{row_index:06d}:{digest}"
        stem = Path(doc_id).stem
        image_dir = dataset_root / "mmlongbench-doc" / "document_images" / stem
        yield {
            "dataset": "mmlongbench-doc",
            "id": str(row_id),
            "doc_id": doc_id,
            "question": question,
            "answer": row["answer"],
            "answer_format": normalize_answer_format(row.get("answer_format")),
            "image_paths": [str(p) for p in list_image_files(image_dir)],
            "evidence_pages": parse_pages(row.get("evidence_pages")),
            "evidence_sources": parse_literal(row.get("evidence_sources"), default=[]),
            "doc_type": row.get("doc_type"),
        }


def _localize_longdocurl_image(dataset_dir: Path, doc_no: str, remote_path: str) -> Path:
    name = Path(remote_path).name
    candidate = dataset_dir / "document_images" / doc_no / name
    if candidate.exists():
        return candidate
    matches = list((dataset_dir / "document_images" / doc_no).glob(name))
    if matches:
        return matches[0]
    return candidate


def load_longdocurl(dataset_root: Path, with_subtask: bool = True) -> Iterable[dict[str, Any]]:
    dataset_dir = dataset_root / "longdocurl"
    filename = "LongDocURL_public_with_subtask_category.jsonl" if with_subtask else "LongDocURL_public.jsonl"
    for row in read_jsonl(dataset_dir / filename):
        doc_no = str(row["doc_no"])
        image_paths = [
            str(_localize_longdocurl_image(dataset_dir, doc_no, image))
            for image in row.get("images", [])
        ]
        yield {
            "dataset": "longdocurl",
            "id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"],
            "answer_format": normalize_answer_format(row.get("answer_format")),
            "image_paths": image_paths,
            "doc_no": doc_no,
            "evidence_pages": parse_pages(row.get("evidence_pages")),
            "evidence_sources": row.get("evidence_sources", []),
            "question_type": row.get("question_type"),
            "task_tag": row.get("task_tag"),
            "subTask": row.get("subTask", []),
        }


def _save_slide_image(value: Any, out_path: Path) -> Path | None:
    if value is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        if value.get("bytes"):
            if not out_path.exists():
                out_path.write_bytes(value["bytes"])
            return out_path
        if value.get("path") and Path(value["path"]).exists():
            return Path(value["path"])
    if isinstance(value, (bytes, bytearray)):
        if not out_path.exists():
            out_path.write_bytes(bytes(value))
        return out_path
    return None


def load_slidevqa(dataset_root: Path, split: str, image_cache: Path) -> Iterable[dict[str, Any]]:
    data_dir = dataset_root / "slidevqa" / "data"
    for row in parquet_rows(data_dir.glob(f"{split}-*.parquet")):
        qa_id = str(row["qa_id"])
        deck_identity = str(row.get("deck_url") or row.get("deck_name") or qa_id)
        deck_cache_key = hashlib.sha1(deck_identity.encode("utf-8")).hexdigest()[:20]
        image_paths: list[str] = []
        for idx, col in enumerate(PAGE_COLUMNS, start=1):
            saved = _save_slide_image(
                row.get(col),
                image_cache / split / "decks" / deck_cache_key / f"page_{idx:02d}.png",
            )
            if saved is not None:
                image_paths.append(str(saved))
        yield {
            "dataset": "slidevqa",
            "split": split,
            "id": qa_id,
            "question": row["question"],
            "answer": row["answer"],
            "answer_format": "String",
            "image_paths": image_paths,
            "evidence_pages": parse_pages(row.get("evidence_pages")),
            "deck_name": row.get("deck_name"),
            "deck_url": row.get("deck_url"),
            "arithmetic_expression": row.get("arithmetic_expression"),
        }


def iter_dataset_records(
    dataset: str,
    dataset_root: Path,
    split: str,
    image_cache: Path,
) -> Iterable[dict[str, Any]]:
    if dataset == "mmlongbench-doc":
        yield from load_mmlongbench(dataset_root, split="train")
    elif dataset == "longdocurl":
        yield from load_longdocurl(dataset_root)
    elif dataset == "slidevqa":
        yield from load_slidevqa(dataset_root, split=split, image_cache=image_cache)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def extract_answer_from_text(text: Any) -> str:
    value = str(text or "").strip()
    concise = re.search(r"<concise_answer>(.*?)<concise_answer>", value, re.S | re.I)
    if concise:
        return concise.group(1).strip()
    match = re.search(r"Extracted answer:\s*(.*?)(?:\n|$)", value, re.I)
    if match:
        return match.group(1).strip()
    final = re.search(r"Final answer:\s*(.*?)(?:\n|$)", value, re.I)
    if final:
        return final.group(1).strip()
    return value
