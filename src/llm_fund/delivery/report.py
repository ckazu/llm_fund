"""Markdown report file output (technical-spec.md 2, 7, 8章).

Writes the report-line/daily Markdown content to `reports/YYYY-MM-DD-<kind>.md`.
Webhook delivery (`delivery/webhook.py`) and `--format json` (`delivery/json_out.py`)
are separate S10 concerns; this module only owns the file artifact.
"""

from datetime import date
from pathlib import Path

REPORT_FILENAME_TEMPLATE = "{date}-{kind}.md"


def report_path(output_dir: str | Path, report_date: date, kind: str) -> Path:
    """Return the destination path for a report, without writing it."""
    filename = REPORT_FILENAME_TEMPLATE.format(date=report_date.isoformat(), kind=kind)
    return Path(output_dir) / filename


def write_report(output_dir: str | Path, report_date: date, kind: str, content_md: str) -> Path:
    """Write `content_md` to `<output_dir>/YYYY-MM-DD-<kind>.md`, creating dirs as needed.

    Overwrites any existing file for the same date/kind (idempotent re-run:
    running the same day's report twice replaces rather than duplicates it).
    """
    path = report_path(output_dir, report_date, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content_md, encoding="utf-8")
    return path
