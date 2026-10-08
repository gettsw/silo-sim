"""LLM API client with tenacity retry."""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import httpx
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt

# Connections kept open per endpoint. A new client per call opened (and leaked) one socket per
# request -- 730 on an SSH tunnel during a 160-request N=20 run, until Windows OpenSSH failed with
# "accept: Too many open files". Requests beyond the pool wait for a free connection instead.
MAX_CONNECTIONS = int(os.environ.get("LLM_MAX_CONNECTIONS", "48"))
_lock = threading.Lock()


@lru_cache(maxsize=None)
def _client(api_base: str, api_key: str) -> OpenAI:
    # 180s read: gpt-oss reasons before answering. pool=None: wait as long as needed for a free
    # connection (a busy pool is normal under high concurrency, not an error).
    http = httpx.Client(
        limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_CONNECTIONS),
        timeout=httpx.Timeout(180.0, pool=None),
    )
    return OpenAI(base_url=api_base, api_key=api_key, http_client=http, max_retries=0)


def get_client(api_base: str, api_key: str) -> OpenAI:
    with _lock:   # lru_cache is not atomic on first call from many threads
        return _client(api_base, api_key)


@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> dict:
    """Call LLM API and return content + token counts.

    Returns:
        {"content": str, "input_tokens": int, "output_tokens": int}

    Retries up to 3 attempts with 2-second intervals on any failure.
    """
    response = get_client(api_base, api_key).chat.completions.create(
        model=model,
        messages=messages,
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "content": choice.message.content or "",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }
