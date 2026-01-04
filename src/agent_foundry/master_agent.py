"""MasterAgent for conversational Agent configuration generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel

from .constants import MasterAgentPrompts
from .factory import AgentFactory
from .schemas import AgentSpec, AskClarifyingQuestionsArgs
from .tools.registry import tool_registry

if TYPE_CHECKING:  # pragma: no cover
    from langgraph.graph.state import CompiledStateGraph
else:  # pragma: no cover
    CompiledStateGraph = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


async def _build_system_prompt() -> str:
    """Build the system prompt with tool catalog."""
    tool_catalog = await _format_tool_catalog()
    # Use replace instead of format to avoid conflicts with { in JSON schemas
    return MasterAgentPrompts.SYSTEM_PROMPT.replace("{tool_catalog}", tool_catalog)


async def _format_tool_catalog() -> str:
    """Format available tools for the system prompt using tool registry."""
    try:
        tools = await tool_registry.get_catalog()
    except RuntimeError:
        logger.exception("Failed to load tool catalog from registry.")
        return "No tools available."

    if not tools:
        return "No tools available."

    catalog_lines = []
    for tool_info in tools:
        # tool_registry.get_catalog returns ToolCatalogItem models
        name = getattr(tool_info, "name", "unknown")
        description = getattr(tool_info, "description", "No description")

        catalog_lines.append(f"### {name}")
        catalog_lines.append(f"**Description:** {description}")

        catalog_lines.append("")

    return "\n".join(catalog_lines).strip()


class Context(BaseModel):
    """Context for MasterAgent."""

    agent_spec: AgentSpec | None = None


class MasterAgent:
    """
    Factory-style wrapper for creating a compiled LangGraph React agent.

    Calling `MasterAgent()` returns a `CompiledStateGraph` instance configured with:
    - A system prompt tailored for agent-spec generation
    - Provided tools (default: `ask_clarity_questions`)
    - Backend model selected by `model` ("pro" or "flash")
    """

    @classmethod
    async def build(
        cls,
        tools: list[Any] | None = None,
        model: str = "pro",
    ) -> CompiledStateGraph[Any, Context, Any, Any]:
        resolved_tools = tools or [ask_clarity_questions, save_agent_spec]

        # Reuse existing model mapping from AgentFactory
        llm = AgentFactory.get_model_instance(model)

        system_prompt = await _build_system_prompt()

        graph = create_agent(
            name="master_agent",
            model=llm,
            tools=resolved_tools,
            system_prompt=system_prompt,
            context_schema=Context,
        )

        return graph


@tool(
    args_schema=AskClarifyingQuestionsArgs,
    return_direct=True,
    description="""向用户提出澄清问题，用于收集创建 AI Agent 所需的详细信息。

使用时机：
- 任何必需字段存在不确定性时
- 工具选择不明确（必须精确匹配可用工具目录）
- 需要确认用户对字段值的偏好
- 需求和描述模糊或不完整

该工具通过结构化问题和建议答案，引导用户提供必要信息。
""",
)
def ask_clarity_questions(
    runtime: ToolRuntime | None = None,
    **kwargs,
) -> str:
    """处理和返回需要向用户提出的澄清问题。

    该函数作为传递函数，接收结构化问题并返回它们，
    由 MasterAgent 向用户展示。

    参数：
        questions: 带建议答案的结构化澄清问题列表

    返回：
        相同的 AskClarifyingQuestionsArgs，用于用户交互
    """
    args = AskClarifyingQuestionsArgs(**kwargs)
    return args.model_dump_json()


@tool(
    args_schema=AgentSpec,
    return_direct=True,
    description="保存 Agent 配置。",
)
def save_agent_spec(runtime: ToolRuntime[Context], **kwargs) -> str:
    """保存 Agent 配置。"""
    spec = AgentSpec(**kwargs)

    # Get context from runtime or fallback to config
    context = runtime.context
    if context is None:
        context = runtime.config.get("configurable", {}).get("context")

    if context:
        context.agent_spec = spec

    return "success"
