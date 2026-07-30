from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

import fetch_datasets


class PdfRenderingTests(unittest.TestCase):
    def test_ligature_warnings_are_filtered_but_other_messages_remain(self) -> None:
        stderr = "\n".join(
            [
                'Syntax Warning: Could not parse ligature component "up" of "angle_up" in parseCharName',
                'Syntax Warning: Could not parse ligature component "down" of "angle_down" in parseCharName',
                'Syntax Warning: Could not parse ligature component "google" of "google_plus" in parseCharName',
                "Syntax Error: Invalid XRef entry",
            ]
        )

        visible, suppressed = fetch_datasets.filter_poppler_stderr(stderr)

        self.assertEqual(suppressed, 3)
        self.assertEqual(visible, ["Syntax Error: Invalid XRef entry"])

    def test_successful_render_summarizes_warnings_and_marks_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "Campaign_038_Introducing_AC_Whitepaper_v5e.pdf"
            pdf.write_bytes(b"%PDF-test")
            images_root = root / "document_images"

            def fake_run(command, **kwargs):
                prefix = Path(command[-1])
                (prefix.parent / f"{prefix.name}-1.png").write_bytes(b"png")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="",
                    stderr=(
                        'Syntax Warning: Could not parse ligature component "up" '
                        'of "angle_up" in parseCharName\n'
                    ),
                )

            stdout = io.StringIO()
            with mock.patch.object(fetch_datasets.subprocess, "run", side_effect=fake_run):
                with redirect_stdout(stdout):
                    fetch_datasets.render_pdf("pdftoppm", pdf, images_root, dpi=144)

            out_dir = images_root / pdf.stem
            self.assertTrue((out_dir / "page-1.png").exists())
            self.assertTrue((out_dir / ".render_complete").exists())
            self.assertIn("suppressed 1 non-fatal", stdout.getvalue())

    def test_success_without_pages_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "empty.pdf"
            pdf.write_bytes(b"%PDF-test")
            result = subprocess.CompletedProcess(
                ["pdftoppm"],
                0,
                stdout="",
                stderr="Syntax Warning: unrelated warning\n",
            )

            with mock.patch.object(fetch_datasets.subprocess, "run", return_value=result):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "rendered no PNG pages"):
                        fetch_datasets.render_pdf("pdftoppm", pdf, root / "images", dpi=144)

            self.assertFalse((root / "images" / "empty" / ".render_complete").exists())


if __name__ == "__main__":
    unittest.main()
