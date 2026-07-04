"""Absolute upper bounds for risk limits (technical-spec.md 6章).

These constants are never relaxable via `config/*.yaml`: `config.py` validates
configured `limits.*` against them at startup and refuses to start (exit code
3, technical-spec.md 9章) if exceeded. Only these constants are defined here
for S3; the full HARD/SOFT rule set (StopLossRequired, MaxLossPerTrade,
CashSufficiency, ...) and `gate.py` are implemented in S5.
"""

ABSOLUTE_MAX_LOSS_PER_TRADE_PCT = 3.0
ABSOLUTE_MAX_POSITION_PCT = 25.0
ABSOLUTE_MAX_TURNOVER_PCT = 50.0
