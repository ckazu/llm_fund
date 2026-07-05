"""プラガブルな LLM バックエンド抽象（technical-spec.md 1章・5章）。

judgment/ と review/ はこのパッケージの `LlmBackend` Protocol 越しにのみ LLM を呼ぶ。
実装は `claude_cli`（Claude Code CLI のサブプロセス実行）と `openai_compat`
（mlx_lm.server / Ollama / LM Studio 等の OpenAI 互換 HTTP サーバ）の2系統で、
用途（role）ごとの使い分けは `router.LlmRouter` が config の `llm.roles` から解決する。
"""
