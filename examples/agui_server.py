"""Run the Agent Foundry AG-UI SSE server."""

import uvicorn

from agent_foundry import create_app


def main():
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()


