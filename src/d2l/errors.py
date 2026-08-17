"""Errors for the D2-L LLM application-consistency reviewer."""

from __future__ import annotations


class D2LError(RuntimeError):
    """Base error for D2-L."""


class D2LDataError(D2LError):
    """Raised when a D2-L data-boundary or Month-7 seal is violated."""


class D2LContractError(D2LError):
    """Raised when a prompt, schema, or freeze invariant is violated."""


class D2LParseError(D2LError):
    """Raised when the LLM output cannot be parsed into the frozen schema."""


class D2LTransportError(D2LError):
    """Raised when the LLM API call fails after retries."""
