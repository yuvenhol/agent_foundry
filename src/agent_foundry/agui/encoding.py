"""Serialization helpers for streaming AG-UI events over SSE."""

from __future__ import annotations

from ag_ui.core.events import BaseEvent
from ag_ui.encoder import EventEncoder

_encoder = EventEncoder()


def sse_encode_event(event: BaseEvent) -> str:
    """Encode an AG-UI event into an SSE 'data: ...\\n\\n' frame."""
    return _encoder.encode(event)
