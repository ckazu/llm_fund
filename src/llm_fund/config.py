"""Application settings: `.env` (secrets) + `config/*.yaml` (non-secret) merged
(technical-spec.md 9章).

`limits.*` are validated against the absolute upper bounds in
`validator/rules.py` at load time. Exceeding them raises `ConfigError`;
`cli.py` maps that to exit code 3 (設定異常).
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_fund.domain.models import MAX_DAILY_TICKET_SEQUENCE
from llm_fund.judgment.client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_SAMPLES,
    DEFAULT_TEMPERATURE,
)
from llm_fund.validator.rules import (
    ABSOLUTE_MAX_POSITION_PCT,
    ABSOLUTE_MAX_TURNOVER_PCT,
)

DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_DB_PATH = "llm_fund.db"
# 営業日カレンダーを持たないための近似既定値。data/loader.py の既定と合わせる。
DEFAULT_MAX_STALENESS_DAYS = 4

# --- tracking（仮想執行・ベンチマーク）既定値（technical-spec.md 7章）------------
# 全戦略（fund / 対照群）に同一適用する開始資本・コスト条件。比較の公平性のため
# ベンチマーク側にも仮想執行と同じ手数料・スリッページを課す。
DEFAULT_STARTING_CAPITAL = 1_000_000.0
DEFAULT_COMMISSION_RATE = 0.0005  # 約定代金の0.05%
DEFAULT_MIN_COMMISSION = 0.0
DEFAULT_SLIPPAGE_PCT = 0.001  # 0.1%
# ランダム対照群のシード（固定して再現可能にする。requirements FR-5）。
DEFAULT_RANDOM_SEED = 42


class ConfigError(Exception):
    """Raised when configuration is missing/invalid or exceeds absolute limits."""


class LLMSettings(BaseModel):
    model: str
    ratio_only: bool = True


class JudgmentSettings(BaseModel):
    """自己一致性・LLM 呼び出しパラメータ（technical-spec.md 5章。任意。既定で 3 サンプル）。"""

    n_samples: int = DEFAULT_N_SAMPLES
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @model_validator(mode="after")
    def _check_positive(self) -> "JudgmentSettings":
        if self.n_samples < 1:
            raise ConfigError(f"judgment.n_samples={self.n_samples} must be >= 1")
        if self.max_tokens < 1:
            raise ConfigError(f"judgment.max_tokens={self.max_tokens} must be >= 1")
        return self


class LimitsSettings(BaseModel):
    max_position_pct: float
    max_turnover_pct: float
    max_instructions_per_day: int
    require_stop_loss: bool = True

    @model_validator(mode="after")
    def _check_absolute_caps(self) -> "LimitsSettings":
        if self.max_position_pct > ABSOLUTE_MAX_POSITION_PCT:
            raise ConfigError(
                f"limits.max_position_pct={self.max_position_pct} exceeds absolute "
                f"cap {ABSOLUTE_MAX_POSITION_PCT}"
            )
        if self.max_turnover_pct > ABSOLUTE_MAX_TURNOVER_PCT:
            raise ConfigError(
                f"limits.max_turnover_pct={self.max_turnover_pct} exceeds absolute "
                f"cap {ABSOLUTE_MAX_TURNOVER_PCT}"
            )
        if self.max_instructions_per_day < 1:
            raise ConfigError(
                f"limits.max_instructions_per_day={self.max_instructions_per_day} must be >= 1"
            )
        if self.max_instructions_per_day > MAX_DAILY_TICKET_SEQUENCE:
            raise ConfigError(
                f"limits.max_instructions_per_day={self.max_instructions_per_day} exceeds the "
                f"daily ticket capacity {MAX_DAILY_TICKET_SEQUENCE}（ticket_no は2桁連番）"
            )
        return self


class BenchmarkSettings(BaseModel):
    index_symbol: str
    momentum_lookback_days: int


class TrackingSettings(BaseModel):
    """仮想執行エンジン・対照群ベンチマーク共通のコスト/資金条件（technical-spec.md 7章）。"""

    starting_capital: float = DEFAULT_STARTING_CAPITAL
    commission_rate: float = DEFAULT_COMMISSION_RATE
    min_commission: float = DEFAULT_MIN_COMMISSION
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT
    random_seed: int = DEFAULT_RANDOM_SEED

    @model_validator(mode="after")
    def _check_non_negative(self) -> "TrackingSettings":
        if self.starting_capital <= 0:
            raise ConfigError(
                f"tracking.starting_capital={self.starting_capital} must be > 0"
            )
        for name in ("commission_rate", "min_commission", "slippage_pct"):
            value = getattr(self, name)
            if value < 0:
                raise ConfigError(f"tracking.{name}={value} must be >= 0")
        return self


class ReportSettings(BaseModel):
    output_dir: str


class DataSettings(BaseModel):
    max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS


class UniverseInstrumentConfig(BaseModel):
    symbol: str
    name: str


class UniverseConfig(BaseModel):
    market: str
    report: bool = True
    trade: bool = False
    cadence: str = "daily"
    instruments: list[UniverseInstrumentConfig] = Field(default_factory=list)


class AppSettings(BaseSettings):
    """Merged application configuration.

    Secrets (`anthropic_api_key`, `notify_webhook_url`) come from `.env`;
    everything else comes from `config/default.yaml` + `config/universes.yaml`.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    anthropic_api_key: str | None = None
    notify_webhook_url: str | None = None
    db_path: str = DEFAULT_DB_PATH

    llm: LLMSettings
    limits: LimitsSettings
    benchmark: BenchmarkSettings
    report: ReportSettings
    data: DataSettings = Field(default_factory=DataSettings)
    judgment: JudgmentSettings = Field(default_factory=JudgmentSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    universes: dict[str, UniverseConfig]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    env_file: Path = DEFAULT_ENV_FILE,
) -> AppSettings:
    """Load and validate settings from `.env` + `config/*.yaml`.

    Raises `ConfigError` if required config is missing/malformed, or `limits`
    exceed the absolute caps in `validator/rules.py`.
    """
    default_conf = _load_yaml(config_dir / "default.yaml")
    universes_conf = _load_yaml(config_dir / "universes.yaml").get("universes", {})
    merged = {**default_conf, "universes": universes_conf}

    try:
        return AppSettings(
            _env_file=env_file if env_file.exists() else None,  # type: ignore[call-arg]
            **merged,
        )
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(str(exc)) from exc
