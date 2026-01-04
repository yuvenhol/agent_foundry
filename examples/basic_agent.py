"""Basic agent example."""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agent_foundry import AgentFactory


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")

    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def main():
    """Run basic agent example."""
    # Create an agent with the calculator tool
    agent = AgentFactory.create_agent(
        model="flash",
        tools=[calculator],
        system_prompt="You are a helpful assistant that can perform calculations.",
    )

    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content="What is 25 * 4 + 10?")]})

    print("Agent response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
