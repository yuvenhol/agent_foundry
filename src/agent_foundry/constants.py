"""Constants for agent_foundry."""


class ModelNames:
    """Default model names."""

    # Can be overridden by environment variables
    DEFAULT_PRO_MODEL = "gpt-5.2"
    DEFAULT_FLASH_MODEL = "gpt-5-mini"


class ModelTypes:
    """Model type identifiers."""

    PRO = "pro"
    FLASH = "flash"


class Defaults:
    """Default values."""

    # Temperature defaults
    DEFAULT_TEMPERATURE = 0.5
    DEFAULT_PRO_TEMPERATURE = 0.7
    DEFAULT_FLASH_TEMPERATURE = 0.5
    DEFAULT_SUBAGENT_TEMPERATURE = 0.5
    DEFAULT_SUMMARIZATION_TEMPERATURE = 0.3

    # Summarization thresholds (if middleware is re-enabled)
    DEFAULT_TRIGGER_TOKENS = 4000
    DEFAULT_TRIGGER_MESSAGES = 10
    DEFAULT_KEEP_MESSAGES = 20

    # Token estimation (rough approximation)
    TOKENS_PER_CHAR_RATIO = 4  # 1 token ≈ 4 characters


class MasterAgentPrompts:
    """Prompts for MasterAgent."""

    SYSTEM_PROMPT = """
## 角色
你是 AI Agent 架构师，通过对话理解用户需求，设计 Agent 方案。


## 配置字段
- **name**: 名称（如 `weather_assistant`）
- **description**: 一句话中文描述
- **system_prompt**: 将角色/风格/限制转化为结构化提示词
- **model**: 默认 "pro"，简单任务用 "flash"
- **temperature**: 默认 0.7，精准任务用低值，创意任务用高值
- **tools**: 从工具目录精确匹配

## 可用工具目录
{tool_catalog}

"""
