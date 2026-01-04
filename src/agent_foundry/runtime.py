"""Runtime for executing LangChain agents based on AgentSpec.

This module mirrors the `MasterAgent` design by exposing a `Runtime` factory
class that builds and returns a compiled LangGraph agent (`CompiledStateGraph`)
from an `AgentSpec`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .factory import AgentFactory
from .schemas import AgentSpec, RuntimeContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from langgraph.graph.state import CompiledStateGraph
else:  # pragma: no cover
    CompiledStateGraph = Any  # type: ignore[misc,assignment]


class Runtime:
    """
    Delivery layer for executing LangGraph agents based on `AgentSpec`.

    While `AgentFactory` handles the low-level assembly of components (LLMs, tools, prompts),
    `Runtime` is responsible for "packaging" the agent for execution.

    Responsibilities:
    1. Orchestration: Calling AgentFactory to assemble the agent.
    2. Compilation: Handling LangGraph's `.compile()` if necessary (often done in Factory currently).
    3. Runtime Config: Injecting memory (Checkpointers), recursion limits, and middleware.
    4. Execution Interface: Providing a consistent async `build` entry point.

    Example:

    ```python
    graph = await Runtime.build(agent_spec, all_available_tools={})
    state = await graph.ainvoke({"messages": [HumanMessage(content="hello")]})
    ```
    """

    @classmethod
    async def build(
        cls,
        agent_spec: AgentSpec,
        all_available_tools: dict[str, Any],
    ) -> CompiledStateGraph[Any, RuntimeContext, Any, Any]:
        """
        Build a production-ready compiled LangGraph agent from an `AgentSpec`.

        This is the primary entry point for turning a declarative spec into
        a runnable agent.

        Args:
            agent_spec: Agent specification model.
            all_available_tools: Dictionary mapping tool name to tool instance.

        Returns:
            Compiled LangGraph agent graph ready for invocation.
        """
        logger.info("Building runtime agent '%s' from spec", agent_spec.name)

        # 1. Assembly: Delegate to AgentFactory to construct the underlying agent components.
        agent = AgentFactory.assemble_agent(
            agent_spec=agent_spec,
            all_available_tools=all_available_tools,
        )

        # 2. Runtime Packaging: In the future, we can wrap the agent with:
        #    - Custom checkpointers for persistence
        #    - SummarizationMiddleware for long-context management
        #    - Execution hooks for monitoring/logging

        # Treat the created agent as a compiled state graph for typing purposes.
        return agent  # type: ignore[return-value]
