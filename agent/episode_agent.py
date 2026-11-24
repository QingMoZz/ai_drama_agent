"""Agent 装配层：组合模型、工具、提示词与调用限制，只构建不执行。"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from agent.episode_tools import EpisodeToolContext, get_episode_summary
from agent.knowledge_tools import search_knowledge

EPISODE_AGENT_SYSTEM_PROMPT = (
    "你是 AI 漫剧创作助手，负责帮助用户了解已保存的剧集内容，"
    "并依据创作知识库回答创作问题。"
    "当用户询问已保存剧集的事实性信息时，必须调用查询工具获取依据，不要凭印象作答。"
    "如果用户没有给出明确的集 ID，先向用户询问，不要自行猜测。"
    "用户未指定版本时查询最新版本，并在回答中说明实际读取的版本号。"
    "当用户要求依据知识库回答时，调用 search_knowledge 检索参考资料。"
    "引用知识库片段时，标注工具返回的 chunk_id；资料不足时如实说明，不要编造。"
    "工具没有找到匹配记录时，如实告知用户，不要编造内容。"
    "回答已保存剧集时只能依据工具返回的字段：不要补写未获取的镜头细节、画面内容或台词。"
    "用户在一句话里问了多项内容时，按顺序逐项回答。"
    "对于一般的创作问题，可以直接回答，无需调用工具。"
)


def build_episode_agent(
    model: BaseChatModel,
    *,
    max_model_calls: int = 4,
    max_tool_calls: int = 4,
) -> CompiledStateGraph:
    """构建剧集查询 Agent，返回编译后的图。

    ``model`` 由调用方创建后传入；``max_model_calls`` 与 ``max_tool_calls``
    分别限制单次 Agent 调用中的模型调用次数与工具调用次数，必须大于等于 1，
    默认值 4 用于首版小规模查询验收。``context_schema`` 只声明上下文的类型，
    具体的数据库路径与知识索引提供器由调用方在执行时通过
    ``EpisodeToolContext`` 传入。工厂本身不执行模型、不访问数据库，只负责装配。
    """
    if max_model_calls < 1:
        raise ValueError(f"max_model_calls 必须大于等于 1，当前为：{max_model_calls}")
    if max_tool_calls < 1:
        raise ValueError(f"max_tool_calls 必须大于等于 1，当前为：{max_tool_calls}")

    middleware = [
        ModelCallLimitMiddleware(
            run_limit=max_model_calls,
            exit_behavior="error",
        ),
        ToolCallLimitMiddleware(
            run_limit=max_tool_calls,
            exit_behavior="error",
        ),
    ]

    return create_agent(
        model=model,
        tools=[get_episode_summary, search_knowledge],
        system_prompt=EPISODE_AGENT_SYSTEM_PROMPT,
        context_schema=EpisodeToolContext,
        middleware=middleware,
    )
