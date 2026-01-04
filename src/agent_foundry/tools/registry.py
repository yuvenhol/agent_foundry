"""Central tool registry for agent_foundry."""

import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolCatalogItem(BaseModel):
    """Schema for a single tool entry in the catalog."""

    name: str
    description: str


class ToolRegistry:
    """Global registry for LangChain tools."""

    _instance: "ToolRegistry | None" = None
    _tools: dict[str, BaseTool] = {}

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: BaseTool instance to register
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> BaseTool:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            BaseTool instance

        Raises:
            ValueError: If tool is not found
        """
        if name not in self._tools:
            available = list(self._tools.keys())
            raise ValueError(f"Tool '{name}' not found. Available tools: {available}")
        return self._tools[name]

    def get_all(self) -> dict[str, BaseTool]:
        """Get all registered tools.

        Returns:
            Dictionary mapping tool names to tool instances
        """
        return self._tools.copy()

    async def get_catalog(self) -> list[ToolCatalogItem]:
        """Get tool catalog for MasterAgent.

        Returns:
            List of ToolCatalogItem objects
        """
        return [
            ToolCatalogItem(name=t.name, description=t.description) for t in self._tools.values()
        ]


# Global singleton instance
tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> BaseTool:
    """Decorator to register a tool.

    Args:
        tool: BaseTool instance to register

    Returns:
        The same tool instance

    Example:
        ```python
        from langchain_core.tools import tool
        from agent_foundry.tools import register_tool

        @register_tool
        @tool
        def my_tool(query: str) -> str:
            '''My custom tool.'''
            return f"Result: {query}"
        ```
    """
    tool_registry.register(tool)
    return tool
