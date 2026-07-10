"""Tests for llm/structured.py (JSON-only output instruction + extraction)."""

import json

import pytest

from llm_fund.llm.structured import extract_json, json_output_instruction

PAYLOAD = {"schema_version": 1, "market_view": "レンジ", "orders": []}


class TestJsonOutputInstruction:
    def test_embeds_schema_in_tag(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        instruction = json_output_instruction(schema)
        assert "<json_schema>" in instruction and "</json_schema>" in instruction
        assert json.dumps(schema, ensure_ascii=False) in instruction
        assert "JSON" in instruction


class TestExtractJson:
    def test_bare_json(self) -> None:
        assert extract_json(json.dumps(PAYLOAD, ensure_ascii=False)) == PAYLOAD

    def test_json_fence(self) -> None:
        text = f"```json\n{json.dumps(PAYLOAD, ensure_ascii=False)}\n```"
        assert extract_json(text) == PAYLOAD

    def test_plain_fence(self) -> None:
        text = f"```\n{json.dumps(PAYLOAD, ensure_ascii=False)}\n```"
        assert extract_json(text) == PAYLOAD

    def test_fence_with_surrounding_prose(self) -> None:
        text = (
            "以下が本日の判断です。\n"
            f"```json\n{json.dumps(PAYLOAD, ensure_ascii=False)}\n```\n"
            "以上です。"
        )
        assert extract_json(text) == PAYLOAD

    def test_bare_json_with_surrounding_prose(self) -> None:
        text = f"判断: {json.dumps(PAYLOAD, ensure_ascii=False)} 以上"
        assert extract_json(text) == PAYLOAD

    def test_non_json_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_json("判断できません")

    def test_json_array_raises(self) -> None:
        # トップレベルがオブジェクトでない JSON は不正として扱う。
        with pytest.raises(ValueError):
            extract_json("[1, 2, 3]")
