"""Data models for LangChain Agent Foundry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RuntimeContext(BaseModel):
    """Runtime Context for agent execution.

    Runtime Context stores static configuration data that persists across
    conversation turns. This includes agent specification, tool registry,
    and execution settings.

    According to LangChain documentation, Runtime Context is appropriate for:
    - Static configuration (AgentSpec)
    - User IDs and authentication data
    - API keys and connections
    - Environment settings

    This data remains constant throughout an agent's lifecycle and does not
    change dynamically like State data (messages, temporary results, etc.).
    """

    agent_spec: AgentSpec = Field(description="Agent specification configuration")
    all_available_tools: dict[str, Any] = Field(
        description="Available tools registry {name: tool_instance}"
    )
    enable_summarization: bool = Field(
        default=True, description="Whether summarization middleware is enabled"
    )
    agent_name: str | None = Field(default=None, description="Agent name extracted from spec")
    agent_model: str | None = Field(
        default=None, description="Agent model type extracted from spec"
    )
    execution_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )


class SubAgentSpec(BaseModel):
    """SubAgent specification (simplified version)."""

    name: str = Field(description="SubAgent name")
    description: str = Field(description="SubAgent responsibility description")
    system_prompt: str = Field(description="SubAgent system prompt")
    tools: list[str] = Field(description="List of tool names")
    model: str = Field(description="Model type: pro or flash")
    temperature: float = Field(default=0.5, ge=0.0, le=2.0, description="Sampling temperature")


class AgentSpec(BaseModel):
    """Main Agent specification."""

    name: str = Field(description="Agent name (English identifier)")
    description: str = Field(description="Agent responsibility description")
    system_prompt: str = Field(description="System prompt")
    model: str = Field(description="Model type: pro or flash")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    tools: list[str] = Field(description="List of tool names")
    subagents: list[SubAgentSpec] | None = Field(
        default=None, description="SubAgent configurations"
    )
    version: str = Field(default="1.0", description="Configuration version")
    max_iterations: int | None = Field(default=None, description="Maximum iterations")


class ClarifyingQuestion(BaseModel):
    """Clarifying question with suggested answers for user requirements gathering."""

    question: str = Field(description="The clarifying question to ask the user")
    suggested_answers: list[str] = Field(
        description="List of suggested answer options for the user to choose from"
    )
    allow_multiple: bool = Field(
        default=False,
        description=(
            "Whether the user can select multiple answers (multi-select). "
            "False means single-choice only."
        ),
    )


class AskClarifyingQuestionsArgs(BaseModel):
    """Arguments for AskClarifyingQuestions tool."""

    questions: list[ClarifyingQuestion] = Field(
        description="List of clarifying questions to ask the user"
    )


class ChatUserMessage(BaseModel):
    """用户消息对象形式，包含消息id与内容"""

    id: str | None = Field(default=None, description="消息id")
    content: str = Field(description="文本内容")


class CommonChatReq(BaseModel):
    sessionId: str | None = Field(default=None, description="会话id，null 表示创建会话")
    message: ChatUserMessage = Field(description="用户消息对象")
