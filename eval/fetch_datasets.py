#!/usr/bin/env python3
"""Download requested document VQA datasets and extract document images."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


DATASETS = {
    "mmlongbench-doc": "yubo2333/MMLongBench-Doc",
    "longdocurl": "dengchao/LongDocURL",
    "slidevqa": "NTT-hil-insight/SlideVQA",
}

IGNORE_PATTERNS = {
    # These two upstream entries currently fail HF consistency checks because
    # the resolved object size does not match the advertised file size.
    "mmlongbench-doc": [
        "documents/dr-vorapptchapter1emissionsources-121120210508-phpapp02_95.pdf",
        "documents/mi_phone.pdf",
    ],
}

MMLONGBENCH_IMAGE_FALLBACKS = [
    "dr-vorapptchapter1emissionsources-121120210508-phpapp02_95",
    "mi_phone",
]

DOCUMENT_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")
POPPLER_LIGATURE_WARNING = re.compile(
    r'^(?:Syntax\s+)?Warning:\s+Could not parse ligature component '
    r'".*" of ".*" in parseCharName\s*$',
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset", help="Dataset output root.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Dataset names to fetch/process. Defaults to all datasets.",
    )
    parser.add_argument("--dpi", type=int, default=144, help="PDF render DPI.")
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Download and unpack archives, but do not render PDFs.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip primary Hugging Face snapshots and process existing raw files under --root.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--render-workers", type=int, default=4)
    return parser.parse_args()


def download(repo_id: str, target: Path, max_workers: int, ignore_patterns: list[str] | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for dataset downloads. "
            "Install eval/requirements-benchmark.txt first."
        ) from exc
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target,
        max_workers=max_workers,
        ignore_patterns=ignore_patterns,
    )


def download_mmlongbench_fallback_images(dataset_dir: Path) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for MMLongBench fallback images. "
            "Install eval/requirements-benchmark.txt first."
        ) from exc
    api = HfApi()
    repo_id = "YeMoKoo/MMLongBench_doc"
    remote_prefix = "MMLongBench_Images_JPG_valid"
    images_root = dataset_dir / "document_images"
    info = api.dataset_info(repo_id)
    for stem in MMLONGBENCH_IMAGE_FALLBACKS:
        out_dir = images_root / stem
        marker = out_dir / ".render_complete"
        if marker.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{remote_prefix}/{stem}_"
        files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(prefix))
        if not files:
            raise FileNotFoundError(f"No fallback images found for {stem}")
        print(f"[fallback-images] {repo_id}:{prefix} -> {out_dir}", flush=True)
        for remote_file in files:
            local = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_file,
                    local_dir=dataset_dir / "_fallback_mmlongbench_images",
                )
            )
            page_name = "page-" + local.stem.removeprefix(f"{stem}_") + local.suffix
            shutil.copy2(local, out_dir / page_name)
        marker.touch()


def extract_archives(dataset_dir: Path) -> None:
    for archive in sorted(dataset_dir.rglob("*")):
        if not archive.is_file() or not archive.name.endswith(ARCHIVE_SUFFIXES):
            continue
        out_dir = archive.parent / archive.name.removesuffix(".tar.gz").removesuffix(".tgz").removesuffix(".tar")
        marker = out_dir / ".extract_complete"
        if marker.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[extract] {archive} -> {out_dir}", flush=True)
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir, filter="data")
        marker.touch()


def find_pdftoppm() -> str:
    candidates = [
        shutil.which("pdftoppm"),
        "/Users/crosshe/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pdftoppm",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("pdftoppm was not found; install poppler or put pdftoppm on PATH.")


def filter_poppler_stderr(stderr: str) -> tuple[list[str], int]:
    """Return actionable Poppler messages and a count of known glyph warnings."""
    visible: list[str] = []
    suppressed = 0
    for line in stderr.splitlines():
        if POPPLER_LIGATURE_WARNING.fullmatch(line.strip()):
            suppressed += 1
        elif line.strip():
            visible.append(line)
    return visible, suppressed


def render_pdf(pdftoppm: str, pdf: Path, images_root: Path, dpi: int) -> None:
    out_dir = images_root / pdf.stem
    marker = out_dir / ".render_complete"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    print(f"[render] {pdf} -> {out_dir}", flush=True)
    result = subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf), str(prefix)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    visible, suppressed = filter_poppler_stderr(result.stderr or "")
    if suppressed:
        print(
            f"[render-warning] {pdf.name}: suppressed {suppressed} "
            "non-fatal malformed ligature-name warning(s)",
            flush=True,
        )
    if visible:
        print("\n".join(visible), file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    pages = sorted(out_dir.glob("page-*.png"))
    if not pages:
        raise RuntimeError(f"pdftoppm returned success but rendered no PNG pages for {pdf}")
    marker.touch()


def render_documents(dataset_dir: Path, dpi: int, render_workers: int) -> None:
    pdftoppm = find_pdftoppm()
    images_root = dataset_dir / "document_images"
    images_root.mkdir(parents=True, exist_ok=True)
    docs: list[Path] = []
    for doc in sorted(dataset_dir.rglob("*")):
        if not doc.is_file() or doc.suffix.lower() not in DOCUMENT_SUFFIXES:
            continue
        if "document_images" in doc.parts:
            continue
        if (images_root / doc.stem / ".render_complete").exists():
            continue
        docs.append(doc)
    if not docs:
        return
    print(f"[render] {len(docs)} PDFs pending in {dataset_dir} using {render_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, render_workers)) as executor:
        futures = [executor.submit(render_pdf, pdftoppm, doc, images_root, dpi) for doc in docs]
        for future in as_completed(futures):
            future.result()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def infer_doc_stem_from_image(image: Path, png_root: Path) -> str:
    stem = image.stem
    for pattern in (r"[_-]page[_-]?\d+$", r"[_-]p\d+$", r"[_-]\d+$"):
        stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
    return stem


def organize_longdocurl_pngs(dataset_dir: Path) -> bool:
    roots = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("png_files")]
    if not roots:
        return False
    images_root = dataset_dir / "document_images"
    images_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for png_root in roots:
        for image in sorted(png_root.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            doc_stem = infer_doc_stem_from_image(image, png_root)
            out_dir = images_root / doc_stem
            out_dir.mkdir(parents=True, exist_ok=True)
            link_or_copy(image, out_dir / image.name)
            count += 1
    print(f"[organize-images] linked/copied {count} LongDocURL images -> {images_root}", flush=True)
    return count > 0


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    for name in args.datasets:
        repo_id = DATASETS[name]
        dataset_dir = root / name
        if args.process_only:
            print(f"[process-only] {dataset_dir}", flush=True)
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Missing local dataset directory: {dataset_dir}")
        else:
            print(f"[download] {repo_id} -> {dataset_dir}", flush=True)
            download(repo_id, dataset_dir, args.max_workers, IGNORE_PATTERNS.get(name))
        extract_archives(dataset_dir)
        if not args.skip_render:
            used_existing_images = name == "longdocurl" and organize_longdocurl_pngs(dataset_dir)
            if not used_existing_images:
                render_documents(dataset_dir, args.dpi, args.render_workers)
            if name == "mmlongbench-doc":
                download_mmlongbench_fallback_images(dataset_dir)


if __name__ == "__main__":
    main()
