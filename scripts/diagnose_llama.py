"""Raw llama-server tool-call probe; never executes the returned tool."""

from __future__ import annotations

import argparse
import json

import httpx


def probe(base_url: str, model: str, tool_choice: str) -> None:
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Responda usando uma tool quando o estado atual for necessário."},
                {"role": "user", "content": "Quanto de memória RAM estou usando agora?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_system_status",
                        "description": "Lê estado atual do sistema.",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                    },
                }
            ],
            "tool_choice": tool_choice,
            "max_tokens": 256,
        },
        timeout=10,
    )
    response.raise_for_status()
    message = response.json().get("choices", [{}])[0].get("message", {})
    print(
        json.dumps(
            {
                "tool_choice": tool_choice,
                "finish_reason": response.json().get("choices", [{}])[0].get("finish_reason"),
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M")
    parser.add_argument("--forced", action="store_true")
    args = parser.parse_args()
    probe(args.base_url, args.model, "required" if args.forced else "auto")
