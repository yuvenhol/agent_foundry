
import json

import httpx
import pytest

import agent_foundry.server as server_mod
from agent_foundry import create_app
from agent_foundry.schemas import AgentSpec





@pytest.mark.asyncio
async def test_e2e_agui_sse(monkeypatch: pytest.MonkeyPatch):
    """End-to-end: consume AG-UI SSE for master + runtime endpoints."""


   
