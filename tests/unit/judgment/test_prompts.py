"""Tests for judgment prompt assembly (technical-spec.md 5章)."""

from llm_fund.judgment.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    format_portfolio_ratio,
)


class TestFormatPortfolioRatio:
    def test_ratios_are_percentages_of_nav(self) -> None:
        summary = format_portfolio_ratio(
            nav=1_000_000.0, cash=250_000.0, holdings={"7203.T": 300_000.0}
        )
        assert "現金比率: 25.0%" in summary
        assert "7203.T: 30.0%" in summary

    def test_no_holdings_states_none(self) -> None:
        summary = format_portfolio_ratio(nav=1_000_000.0, cash=1_000_000.0, holdings={})
        assert "保有ポジション: なし" in summary

    def test_zero_nav_returns_placeholder(self) -> None:
        summary = format_portfolio_ratio(nav=0.0, cash=0.0, holdings={})
        assert "未初期化" in summary

    def test_does_not_leak_absolute_amounts(self) -> None:
        # 実額（250000 等）ではなく比率のみを送る（technical-spec.md 5章 プライバシー）。
        summary = format_portfolio_ratio(
            nav=1_000_000.0, cash=250_000.0, holdings={"7203.T": 300_000.0}
        )
        assert "250000" not in summary
        assert "300000" not in summary


class TestBuildUserPrompt:
    def test_wraps_data_in_xml_tags(self) -> None:
        prompt = build_user_prompt(briefing_md="| 表 |", portfolio_summary="現金比率: 100.0%")
        assert "<briefing>" in prompt and "</briefing>" in prompt
        assert "| 表 |" in prompt
        assert "<portfolio>" in prompt
        assert "現金比率: 100.0%" in prompt

    def test_missing_materials_use_placeholders(self) -> None:
        prompt = build_user_prompt(briefing_md="b", portfolio_summary="p")
        assert "未設定" in prompt  # policy / criteria placeholders
        assert "まだありません" in prompt  # recent instructions placeholder

    def test_materials_injected_when_present(self) -> None:
        prompt = build_user_prompt(
            briefing_md="b",
            portfolio_summary="p",
            policy="現金比率を高めに保つ",
            criteria="IFO 幅は ATR の 1.5 倍",
            recent_instructions="20260703-01 BUY 勝率50%",
        )
        assert "現金比率を高めに保つ" in prompt
        assert "IFO 幅は ATR の 1.5 倍" in prompt
        assert "20260703-01 BUY 勝率50%" in prompt


class TestSystemPrompt:
    def test_declares_data_not_command(self) -> None:
        # プロンプトインジェクション防御の宣言が含まれること。
        assert "データ" in SYSTEM_PROMPT
        assert "命令" in SYSTEM_PROMPT
        assert "submit_judgment" in SYSTEM_PROMPT

    def test_prompt_version_is_nonempty(self) -> None:
        assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION
