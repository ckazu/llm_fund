"""File-output tests for `delivery/report.py` (S4)."""

from datetime import date
from pathlib import Path

from llm_fund.delivery.report import report_path, write_report


class TestReportPath:
    def test_formats_date_and_kind(self, tmp_path: Path) -> None:
        path = report_path(tmp_path, date(2026, 7, 4), "daily")

        assert path == tmp_path / "2026-07-04-daily.md"


class TestWriteReport:
    def test_writes_content_to_expected_path(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports"

        path = write_report(output_dir, date(2026, 7, 4), "report", "# hello\n")

        assert path == output_dir / "2026-07-04-report.md"
        assert path.read_text(encoding="utf-8") == "# hello\n"

    def test_creates_missing_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "nested" / "reports"

        path = write_report(output_dir, date(2026, 7, 4), "daily", "content")

        assert path.exists()

    def test_rerun_overwrites_same_day_report(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports"
        write_report(output_dir, date(2026, 7, 4), "daily", "first")

        path = write_report(output_dir, date(2026, 7, 4), "daily", "second")

        assert path.read_text(encoding="utf-8") == "second"
