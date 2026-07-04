"""Tests for config.py: .env + config/*.yaml merging and limits validation."""

from pathlib import Path

import pytest
import yaml

from llm_fund.config import ConfigError, load_settings

VALID_DEFAULT = {
    "llm": {"model": "claude-sonnet-5", "ratio_only": True},
    "limits": {
        "max_position_pct": 15.0,
        "max_turnover_pct": 30.0,
        "max_instructions_per_day": 5,
        "require_stop_loss": True,
    },
    "benchmark": {"index_symbol": "1306.T", "momentum_lookback_days": 120},
    "report": {"output_dir": "reports"},
}

VALID_UNIVERSES = {
    "universes": {
        "jp_stocks": {
            "market": "jp",
            "report": True,
            "trade": True,
            "cadence": "daily",
            "instruments": [{"symbol": "7203.T", "name": "トヨタ自動車"}],
        }
    }
}


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    d.mkdir()
    _write_yaml(d / "default.yaml", VALID_DEFAULT)
    _write_yaml(d / "universes.yaml", VALID_UNIVERSES)
    return d


@pytest.fixture
def missing_env(tmp_path: Path) -> Path:
    return tmp_path / "missing.env"


class TestLoadSettings:
    def test_loads_valid_config(self, config_dir: Path, missing_env: Path) -> None:
        settings = load_settings(config_dir=config_dir, env_file=missing_env)

        assert settings.llm.model == "claude-sonnet-5"
        assert settings.limits.max_position_pct == 15.0
        assert settings.benchmark.index_symbol == "1306.T"
        assert "jp_stocks" in settings.universes
        assert settings.universes["jp_stocks"].instruments[0].symbol == "7203.T"

    def test_env_secrets_loaded_from_env_file(self, config_dir: Path, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-test\n", encoding="utf-8")

        settings = load_settings(config_dir=config_dir, env_file=env_file)

        assert settings.anthropic_api_key == "sk-test"

    def test_max_position_pct_over_absolute_cap_is_rejected(
        self, config_dir: Path, missing_env: Path
    ) -> None:
        bad = {**VALID_DEFAULT, "limits": {**VALID_DEFAULT["limits"], "max_position_pct": 26.0}}
        _write_yaml(config_dir / "default.yaml", bad)

        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir, env_file=missing_env)

    def test_max_turnover_pct_over_absolute_cap_is_rejected(
        self, config_dir: Path, missing_env: Path
    ) -> None:
        bad = {**VALID_DEFAULT, "limits": {**VALID_DEFAULT["limits"], "max_turnover_pct": 51.0}}
        _write_yaml(config_dir / "default.yaml", bad)

        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir, env_file=missing_env)

    def test_limit_exactly_at_absolute_cap_is_accepted(
        self, config_dir: Path, missing_env: Path
    ) -> None:
        ok = {**VALID_DEFAULT, "limits": {**VALID_DEFAULT["limits"], "max_position_pct": 25.0}}
        _write_yaml(config_dir / "default.yaml", ok)

        settings = load_settings(config_dir=config_dir, env_file=missing_env)

        assert settings.limits.max_position_pct == 25.0

    def test_missing_required_section_raises_config_error(
        self, config_dir: Path, missing_env: Path
    ) -> None:
        incomplete = {k: v for k, v in VALID_DEFAULT.items() if k != "llm"}
        _write_yaml(config_dir / "default.yaml", incomplete)

        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir, env_file=missing_env)

    def test_real_repo_config_files_are_valid(self) -> None:
        """Guards against the actual config/*.yaml drifting out of schema."""
        settings = load_settings(env_file=Path(".env.example"))

        assert settings.limits.max_position_pct <= 25.0
        assert settings.limits.max_turnover_pct <= 50.0
