#!/usr/bin/env python3
"""Minimal DeepSeek API connectivity check (OpenAI-compatible client).

Does not modify A0/A2, governance, budgets, or experiment outputs.
Prints only the model response text and token usage — never the API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_IMPL_ROOT = _SCRIPTS_DIR.parent
_SRC = _IMPL_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openai import OpenAI  # noqa: E402

from deepseek_config import load_deepseek_settings  # noqa: E402


def main() -> int:
    settings = load_deepseek_settings()

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: deepseek-connection-ok",
            }
        ],
        max_tokens=100,
        stream=False,
    )

    content = (response.choices[0].message.content or "").strip()
    usage = response.usage

    print(f"model_response: {content}")
    if usage is None:
        print("token_usage: unavailable")
    else:
        print(
            "token_usage: "
            f"prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"total={usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
