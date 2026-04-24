"""OpenAI-function-calling JSON schemas for the 5 MCP tools.

Used by the fast-path dispatcher (`src/agent/fast_graph.py`) so the LLM can
decide in a single round-trip whether to call a tool or stream a direct reply.

Schema style: OpenAI's `tools=[{"type": "function", "function": {...}}]`
format. Both OpenAI and vLLM's auto-tool-choice (with `hermes` parser) accept
this shape via LiteLLM's unified interface.

Parameters are kept MINIMAL for latency — every extra field is another token
the LLM must emit before we can act. `account_id` is NOT a parameter (the
dispatcher injects it from session state to keep the LLM from hallucinating
it).
"""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_bill",
            "description": (
                "Fetch the caller's recent bills (amount, period, status). "
                "Use when the caller asks about their bill, invoice, amount due, "
                "or payment status. Requires authentication."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many recent bills to return (default 2).",
                        "default": 2,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_plan",
            "description": (
                "Change the caller's subscription plan. Only call AFTER the "
                "caller explicitly confirms the new plan name (e.g. 'yes, "
                "upgrade to scale'). Never call this speculatively."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_plan": {
                        "type": "string",
                        "enum": ["starter", "growth", "scale", "enterprise"],
                        "description": "The target plan name.",
                    },
                },
                "required": ["new_plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_kb",
            "description": (
                "Search the Acme Cloud knowledge base for how-to / policy / "
                "pricing / technical answers. Use when the caller's question "
                "can be answered from public docs (backup retention, region "
                "list, plan limits, connection troubleshooting)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to search the KB with.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Transfer the caller to a human support agent and create a "
                "ticket. Call when the caller asks for a human, when the "
                "request is destructive (account deletion, refund >€100), or "
                "when you're clearly unable to help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "One-line summary of the caller's issue for the ticket.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "critical"],
                        "default": "normal",
                    },
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "authenticate",
            "description": (
                "Verify a caller's identity by their email address. Call when "
                "the caller gives you an email address at the start of the "
                "call, or when they correct an email you got wrong."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The caller's account email.",
                    },
                },
                "required": ["email"],
            },
        },
    },
]

# Names the dispatcher will recognize — any tool_call.function.name outside
# this set is rejected rather than executed.
TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOL_SCHEMAS}
