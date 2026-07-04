"""Shared numeric constants used across layers (no dependencies; safe to import anywhere).

`PERCENT_DIVISOR` is the single source of truth for converting between a percentage
and a ratio (`pct / PERCENT_DIVISOR`) or a ratio and a percentage (`ratio * PERCENT_DIVISOR`).
Defining it once avoids the same literal `100.0` drifting across the validator, gate,
prompts, briefing and tracking layers.
"""

# パーセント <-> 比率 変換の除数（100%）。全レイヤで共有する唯一の定義。
PERCENT_DIVISOR = 100.0
