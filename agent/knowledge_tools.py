"""工具层：向模型暴露知识库的向量检索能力。"""

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, field_validator

from agent.episode_tools import EpisodeToolContext
from agent.knowledge_retrieval import KnowledgeHit, retrieve_knowledge


class KnowledgeSearchInput(BaseModel):
    """工具输入契约：模型只能看到要检索的查询文本。"""

    query: str = Field(description="要在知识库中检索的问题或关键词")

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query 去掉首尾空白后不能为空")
        return query


@tool(args_schema=KnowledgeSearchInput)
def search_knowledge(
    query: str,
    runtime: ToolRuntime[EpisodeToolContext],
) -> list[KnowledgeHit]:
    """检索 AI 漫剧创作知识库，返回前 3 条可追溯的参考片段。

    适用于查询悬疑节奏、线索揭示、分镜连续性、抖音短剧与身份反转钩子等
    创作参考资料。返回按检索顺序排列的列表，每项包含片段正文 text、
    来源 source_id、片段 chunk_id、文档标题 document_title、小节标题
    section_title 与检索分数 score；知识库为空时返回空列表。运行上下文
    未提供知识索引时抛出 RuntimeError。
    """
    provider = runtime.context.knowledge_index_provider
    if provider is None:
        raise RuntimeError(
            "当前运行上下文未提供知识索引，无法检索知识库；"
            "请通过 EpisodeToolContext.knowledge_index_provider 传入提供器"
        )

    index = provider()
    return retrieve_knowledge(index, query)
