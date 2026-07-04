"""`--no-llm` 用の決定論的テンプレート判断（technical-spec.md 2, 5章 / requirements FR 再現性）。

LLM を呼ばずにパイプライン全体（briefing → judgment → validator → delivery）を
課金なしで貫通テストできるようにするためのフォールバック判断。決定論であること
（同一入力で必ず同一出力）が要件なので、テンプレートは常に現状維持（NO_TRADE）を
返し、新規の売買指示は一切出さない。これは技術的スモークテストの土台であって
戦略ではない。将来「単純な保有継続ルール」を足す場合もここに決定論ロジックとして
置く。self-consistency（複数サンプルの多数決）は LLM の不安定性対策であり、
決定論のテンプレートには不要なので実行回数は常に1回（technical-spec.md 5章）。
"""

from llm_fund.domain.models import JudgmentResult

# テンプレート判断が返す固定の市況コメントと NO_TRADE 理由。監査ログ/レポートに
# 「これは LLM 判断ではなくテンプレート」と明示するための文言。
TEMPLATE_MARKET_VIEW = "テンプレート判断（--no-llm）: LLM を呼び出していない"
TEMPLATE_NO_TRADE_REASON = (
    "テンプレート判断（--no-llm）は決定論的に現状維持（NO_TRADE）とする"
)


def template_judgment() -> JudgmentResult:
    """常に NO_TRADE（新規指示なし・現状維持）の決定論的判断を返す。"""
    return JudgmentResult(
        orders=[],
        market_view=TEMPLATE_MARKET_VIEW,
        no_trade=True,
        no_trade_reason=TEMPLATE_NO_TRADE_REASON,
    )
