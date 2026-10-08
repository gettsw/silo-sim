"""Adapter factories that convert provider-specific LLM clients into
the ``LLMFn = Callable[[list[dict]], str]`` interface expected by LLMAgent.

Each factory captures credentials/config in a closure so the returned
callable has no external dependencies — easy to swap, mock, or pickle.

Available adapters
------------------
make_openai_fn(api_base, api_key, model, **kwargs)
    Wraps silo_sim.utils.llm.call_llm.  Works with any OpenAI-compatible
    endpoint (OpenAI, DeepSeek, Grok, Ollama with openai shim, etc.).

make_mock_fn(responses)
    Returns responses from a pre-supplied list, cycling when exhausted.
    Designed for deterministic unit tests — no network required.
"""

from __future__ import annotations

from itertools import cycle
from typing import Any, Callable

LLMFn = Callable[[list[dict[str, str]]], str]


def make_openai_fn(
    api_base: str,
    api_key: str,
    model: str,
    **call_llm_kwargs: Any,
) -> LLMFn:
    """Return an LLMFn backed by silo_sim.utils.llm.call_llm.

    Parameters
    ----------
    api_base:   OpenAI-compatible base URL (e.g. "https://api.openai.com/v1")
    api_key:    API key string
    model:      Model identifier (e.g. "gpt-4o", "deepseek-chat")
    **call_llm_kwargs:
        Extra keyword arguments forwarded to call_llm (currently unused by
        the base implementation, reserved for future extension).

    Returns
    -------
    Callable[[list[dict]], str]
        Accepts an OpenAI message list, returns the assistant content string.
        Token counts are discarded — access them via call_llm directly if
        you need them for cost tracking.
    """
    # Import here so the simulation module has no hard dependency on the
    # SILO-BENCH LLM utils when they are not needed (e.g. pure RuleAgent runs).
    from silo_sim.utils.llm import call_llm

    def llm_fn(messages: list[dict[str, str]]) -> str:
        result = call_llm(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            **call_llm_kwargs,
        )
        return result["content"]

    return llm_fn


def make_mock_fn(responses: list[str], *, loop: bool = True) -> LLMFn:
    """Return an LLMFn that replays a fixed list of response strings.

    Parameters
    ----------
    responses:
        Ordered list of strings the mock will return.
    loop:
        If True (default), cycle through the list indefinitely.
        If False, raise StopIteration after the list is exhausted.

    Usage in tests::

        fn = make_mock_fn(['{"send": [{"to": 1, "content": "hi"}]}'])
        agent = LLMAgent(agent_id=0, llm_fn=fn)
    """
    if not responses:
        raise ValueError("responses list must not be empty")

    if loop:
        _iter = cycle(responses)

        def llm_fn(_messages: list[dict[str, str]]) -> str:
            return next(_iter)

    else:
        _iter_once = iter(responses)

        def llm_fn(_messages: list[dict[str, str]]) -> str:  # type: ignore[misc]
            try:
                return next(_iter_once)
            except StopIteration:
                raise StopIteration(
                    "make_mock_fn exhausted its response list (loop=False)"
                )

    return llm_fn
