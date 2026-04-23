"""
MCP tools surfaced to the LangGraph agent.

Each module exports a callable that:
- takes typed arguments
- returns a small dict ready for inclusion in the LLM prompt
- is registered with the MCP server in `src.agent.tools.server`
"""

from .accounts import change_plan, get_account
from .billing import get_bill
from .escalation import escalate_to_human
from .knowledge import lookup_kb

__all__ = [
    "change_plan",
    "escalate_to_human",
    "get_account",
    "get_bill",
    "lookup_kb",
]
