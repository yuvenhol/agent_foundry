"""SubAgentTool: Wraps a sub-agent as a LangChain BaseTool."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubAgentToolInput(BaseModel):
    """Input schema for SubAgentTool."""

    task: str = Field(description="Task description for the sub-agent to execute")


class SubAgentTool(BaseTool):
    """
    Wraps a sub-agent as a tool that can be called by a parent agent.

    This allows:
    1. Context isolation: Sub-agent's conversation is separate from parent
    2. Independent summarization: Sub-agent can have its own SummarizationMiddleware
    3. Error propagation: Sub-agent errors are passed to the parent
    4. Flexible configuration: Different models, tools, and prompts

    The sub-agent receives a task, executes it with its own tools and configuration,
    and returns the result as a string to the parent agent.
    """

    name: str
    description: str
    subagent: Runnable
    args_schema: type[BaseModel] = SubAgentToolInput

    def __init__(
        self,
        name: str,
        description: str,
        subagent: Runnable,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SubAgentTool.

        Args:
            name: Tool name (unique identifier)
            description: Tool description (what the sub-agent does)
            subagent: Runnable agent instance
        """
        super().__init__(
            name=name,
            description=description,
            subagent=subagent,
            **kwargs,
        )

    def _run(
        self,
        task: str,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the sub-agent with the given task.

        Args:
            task: Task description for the sub-agent
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        Returns:
            Sub-agent execution result as string

        Example:
            ```python
            tool = SubAgentTool(
                name="data_analyzer",
                description="Analyzes data and generates insights",
                subagent=subagent_runnable
            )
            result = tool._run("Analyze this dataset and find patterns")
            ```
        """
        logger.info(f"Sub-agent '{self.name}' executing task: {task[:100]}...")

        try:
            # Prepare input for the sub-agent
            # The sub-agent expects messages in the format: {"messages": [...]}
            input_data = {
                "messages": [HumanMessage(content=task)],
                "agent_scratchpad": [],
            }

            # Run the sub-agent
            result = self.subagent.invoke(input_data)

            # Extract the output
            # The result format depends on how the agent is structured
            # For create_tool_calling_agent, it's typically an AIMessage
            output = self._extract_output(result)

            logger.info(f"Sub-agent '{self.name}' completed task successfully")
            return output

        except Exception as e:
            logger.error(f"Sub-agent '{self.name}' failed: {e}")
            raise Exception(f"Sub-agent '{self.name}' error: {str(e)}") from e

    async def _arun(
        self,
        task: str,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Async version of _run.

        Args:
            task: Task description for the sub-agent
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        Returns:
            Sub-agent execution result as string
        """
        logger.info(f"Sub-agent '{self.name}' (async) executing task: {task[:100]}...")

        try:
            input_data = {
                "messages": [HumanMessage(content=task)],
                "agent_scratchpad": [],
            }

            # Run the sub-agent asynchronously (if supported)
            # Note: Not all agents support async invoke
            if hasattr(self.subagent, "ainvoke"):
                result = await self.subagent.ainvoke(input_data)
            else:
                # Fallback to sync version
                result = self.subagent.invoke(input_data)

            output = self._extract_output(result)

            logger.info(f"Sub-agent '{self.name}' completed task successfully")
            return output

        except Exception as e:
            logger.error(f"Sub-agent '{self.name}' failed: {e}")
            raise Exception(f"Sub-agent '{self.name}' error: {str(e)}") from e

    def _extract_output(self, result: Any) -> str:
        """
        Extract the output string from the sub-agent result.

        Args:
            result: Output from sub-agent.invoke()

        Returns:
            Extracted string output
        """
        # Handle dict results (common from agent executors)
        if isinstance(result, dict):
            return self._extract_from_dict(result)

        # Handle AIMessage directly
        if isinstance(result, AIMessage):
            return str(result.content)

        # Handle string directly
        if isinstance(result, str):
            return result

        # Fallback for unknown types
        logger.warning(f"Unknown result type from sub-agent: {type(result)}")
        return str(result)

    def _extract_from_dict(self, result: dict[str, Any]) -> str:
        """Extract output from a dictionary result."""
        # Priority 1: Direct 'output' key
        if "output" in result:
            return str(result["output"])

        # Priority 2: Last message from 'messages' key
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return str(last_message.content)
            return str(last_message)

        # Priority 3: String representation of the dict
        return str(result)

    def invoke(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, str]:
        """
        Invoke the tool with structured input.

        Args:
            input_data: Dictionary with "task" key
            **kwargs: Additional arguments

        Returns:
            Dictionary with result

        Example:
            ```python
            result = tool.invoke({"task": "Analyze data"})
            print(result["output"])
            ```
        """
        task = input_data.get("task", "")
        output = self._run(task)
        return {"output": output}

    async def ainvoke(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, str]:
        """
        Async invoke the tool with structured input.

        Args:
            input_data: Dictionary with "task" key
            **kwargs: Additional arguments

        Returns:
            Dictionary with result
        """
        task = input_data.get("task", "")
        output = await self._arun(task)
        return {"output": output}


def create_subagent_tool(
    name: str,
    description: str,
    subagent: Runnable,
) -> SubAgentTool:
    """
    Factory function to create a SubAgentTool.

    Args:
        name: Tool name
        description: Tool description
        subagent: Runnable agent instance

    Returns:
        SubAgentTool instance

    Example:
        ```python
        subagent = AgentFactory.create_agent(
            model="flash",
            tools=[tool1, tool2],
            system_prompt="You are a data analyst..."
        )

        tool = create_subagent_tool(
            name="data_analyzer",
            description="Analyzes datasets and generates insights",
            subagent=subagent
        )
        ```
    """
    return SubAgentTool(
        name=name,
        description=description,
        subagent=subagent,
    )
