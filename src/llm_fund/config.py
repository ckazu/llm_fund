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
from llm_fund.validator.rules import (
    ABSOLUTE_MAX_POSITION_PCT,
    ABSOLUTE_MAX_TURNOVER_PCT,
)

DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_DB_PATH = "llm_fund.db"
# 営業日カレンダーを持たないための近似既定値。data/loader.py の既定と合わせる。
DEFAULT_MAX_STALENESS_DAYS = 4


class ConfigError(Exception):
    """Raised when configuration is missing/invalid or exceeds absolute limits."""


class LLMSettings(BaseModel):
    model: str
    ratio_only: bool = True


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
