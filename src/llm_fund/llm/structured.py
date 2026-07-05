"""プロンプト指示による構造化出力の共通部品（technical-spec.md 5章）。

tool use が使えないバックエンド（claude CLI / ローカル LLM）向けに、
「JSON スキーマを明示して JSON のみを出力させる」指示文の組み立てと、
応答テキストからの JSON 抽出（```json フェンス対応）を提供する。
抽出後の pydantic 検証は呼び出し側（judgment / review）の責務。
"""

import json
import re
from collections.abc import Mapping
from typing import Any

# ```json ... ``` / ``` ... ``` フェンスの中身を取り出す（最初のフェンスのみ）。
_FENCE_PATTERN = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)

_INSTRUCTION_TEMPLATE = (
    "出力形式（厳守）: 応答は以下の JSON スキーマに厳密に従う単一の JSON "
    "オブジェクトのみとする。説明文・前置き・後書き・コードフェンス以外の"
    "テキストを一切含めないこと。\n"
    "<json_schema>\n{schema}\n</json_schema>"
)


def json_output_instruction(schema: Mapping[str, Any]) -> str:
    """システムプロンプトに追記する「JSON のみを出力せよ」指示文を組み立てる。"""
    return _INSTRUCTION_TEMPLATE.format(schema=json.dumps(schema, ensure_ascii=False))


def extract_json(text: str) -> dict[str, Any]:
    """応答テキストから JSON オブジェクトを抽出する。

    素の JSON、```json フェンス付き、フェンス前後に説明文が付いた応答の
    いずれにも対応する。抽出・パースできない場合は `ValueError`。
    """
    candidates = [text.strip()]
    fenced = _FENCE_PATTERN.search(text)
    if fenced is not None:
        candidates.insert(0, fenced.group(1).strip())
    # フェンスなしで前後に説明文が付いた場合に備え、最初の '{' から最後の '}' までも試す。
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("応答テキストから JSON オブジェクトを抽出できない")
