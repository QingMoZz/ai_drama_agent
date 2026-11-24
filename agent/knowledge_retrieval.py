"""检索层：在已建好的知识库索引上执行查询并整理可追溯的结果。"""

from typing import TypedDict

from langchain_core.vectorstores import InMemoryVectorStore

KNOWLEDGE_SEARCH_TOP_K = 3


class KnowledgeHit(TypedDict):
    """一条知识库检索结果，字段可直接用于回答引用与溯源。"""

    text: str
    source_id: str
    chunk_id: str
    document_title: str
    section_title: str
    score: float


def retrieve_knowledge(
    index: InMemoryVectorStore,
    query: str,
) -> list[KnowledgeHit]:
    """在已建好的索引中检索前 3 条片段并整理为结果字典。

    按向量库返回的顺序组织结果，``score`` 为余弦相似度并转为 Python
    ``float``；索引为空时返回空列表。索引需由调用方提前建好并传入。
    """
    results = index.similarity_search_with_score(query, k=KNOWLEDGE_SEARCH_TOP_K)
    return [
        {
            "text": document.page_content,
            "source_id": document.metadata.get("source_id", ""),
            "chunk_id": document.metadata.get("chunk_id", ""),
            "document_title": document.metadata.get("document_title", ""),
            "section_title": document.metadata.get("section_title", ""),
            "score": float(score),
        }
        for document, score in results
    ]
