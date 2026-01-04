"""ID helpers for AG-UI runs/messages/tool calls."""

from __future__ import annotations

import uuid


def new_id(prefix: str) -> str:
    """Generate a stable-ish opaque id with a human-friendly prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"
