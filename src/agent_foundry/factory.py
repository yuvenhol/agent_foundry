"""AgentFactory for building LangChain agents from AgentSpec."""

import logging

from langchain.agents import create_agent
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from .config import settings
from .constants import Defaults, ModelTypes
from .llm.factory import get_llm
from .schemas import AgentSpec, SubAgentSpec
from .subagent_tool import SubAgentTool

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating LangChain agents from AgentSpec.

    Responsibilities:
    1. Model factory: Map model strings ("pro"/"flash") to LLM instances
    2. Tool loading: Load tools from global registry
    3. SubAgent handling: Recursively build sub-agents and wrap as tools
    4. Agent assembly: Call create_tool_calling_agent or similar
    """

    @staticmethod
    def get_model_instance(model: str, temperature: float | None = None):
        """
        Get LLM instance from model string.

        Args:
            model: Model type ("pro" or "flash")
            temperature: Optional sampling temperature

        Returns:
            ChatOpenAI instance

        Raises:
            ValueError: If model string is invalid
        """
        if temperature is None:
            temperature = Defaults.DEFAULT_TEMPERATURE

        if model == ModelTypes.PRO:
            return get_llm(settings.pro_model, temperature=temperature)
        elif model == ModelTypes.FLASH:
            return get_llm(settings.flash_model, temperature=temperature)
        else:
            raise ValueError(
                f"Invalid model: '{model}'. Must be '{ModelTypes.PRO}' or '{ModelTypes.FLASH}'."
            )

    @staticmethod
    def _load_tools(tool_names: list[str], all_available_tools: dict) -> list[BaseTool]:
        """
        Load tools from the global registry.

        Args:
            tool_names: List of tool names to load
            all_available_tools: Dictionary of available tools {name: tool}

        Returns:
            List of BaseTool instances

        Raises:
            ValueError: If a tool is not found
        """
        loaded_tools = []
        missing_tools = []

        for tool_name in tool_names:
            if tool_name in all_available_tools:
                loaded_tools.append(all_available_tools[tool_name])
            else:
                missing_tools.append(tool_name)

        if missing_tools:
            available_names = list(all_available_tools.keys())
            raise ValueError(
                f"Tools not found: {missing_tools}. Available tools: {available_names}"
            )

        return loaded_tools

    @staticmethod
    def create_agent(
        model: str,
        tools: list[BaseTool],
        system_prompt: str,
        temperature: float | None = None,
        middleware: list | None = None,
    ) -> Runnable:
        """
        Create a LangChain agent with the given configuration.

        Args:
            model: Model type ("pro" or "flash")
            tools: List of tools for the agent
            system_prompt: System prompt for the agent
            temperature: Optional temperature override
            middleware: Optional list of middleware for custom processing

        Returns:
            Configured Runnable agent

        Example:
            ```python
            agent = AgentFactory.create_agent(
                model="pro",
                tools=[tool1, tool2],
                system_prompt="You are a helpful assistant...",
            )
            result = agent.invoke({"messages": [HumanMessage("Hello")]})
            ```
        """
        # Get model instance
        model_instance = AgentFactory.get_model_instance(model, temperature)

        # Create the agent
        agent = create_agent(
            model=model_instance,
            tools=tools,
            system_prompt=system_prompt,
        )

        # Add middleware if provided
        # Note: Middleware integration can be added here in the future
        if middleware:
            pass

        return agent

    @staticmethod
    def assemble_agent(
        agent_spec: AgentSpec,
        all_available_tools: dict,
    ) -> Runnable:
        """
        Assemble an agent from an AgentSpec.

        This is a low-level assembly method that handles:
        1. Loading tools from registry
        2. Building subagents recursively as tools
        3. Instantiating the underlying agent structure

        Args:
            agent_spec: AgentSpec object or dict
            all_available_tools: Dictionary of available tools {name: tool}

        Returns:
            Configured Runnable agent (uncompiled or compiled depending on backend)

        Raises:
            ValueError: If spec is invalid or tools are missing
        """

        # Step 1: Load main tools
        tools = AgentFactory._load_tools(agent_spec.tools, all_available_tools)

        # Step 2: Build subagents and add them as tools
        if agent_spec.subagents:
            for subagent_spec in agent_spec.subagents:
                subagent_tool = AgentFactory._build_subagent_tool(
                    subagent_spec, all_available_tools
                )
                tools.append(subagent_tool)

        # Step 3: Create the agent
        agent = AgentFactory.create_agent(
            model=agent_spec.model,
            tools=tools,
            system_prompt=agent_spec.system_prompt,
            temperature=agent_spec.temperature,
        )

        logger.info(
            f"Assembled agent '{agent_spec.name}' with {len(tools)} tools, "
            f"model={agent_spec.model}, temperature={agent_spec.temperature}"
        )

        return agent

    @staticmethod
    def _build_subagent_tool(subagent_spec: SubAgentSpec, all_available_tools: dict) -> BaseTool:
        """
        Build a subagent and wrap it as a tool.

        Args:
            subagent_spec: SubAgent specification
            all_available_tools: Dictionary of available tools

        Returns:
            BaseTool that wraps the subagent
        """
        # Create the subagent
        # Note: Subagents don't have nested subagents in current design
        subagent_tools = AgentFactory._load_tools(subagent_spec.tools, all_available_tools)

        subagent = AgentFactory.create_agent(
            model=subagent_spec.model,
            tools=subagent_tools,
            system_prompt=subagent_spec.system_prompt,
            temperature=subagent_spec.temperature,
        )

        # Wrap as a tool
        return SubAgentTool(
            name=subagent_spec.name,
            description=subagent_spec.description,
            subagent=subagent,
        )
