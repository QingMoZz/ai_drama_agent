"""知识库建库层：把切分片段写入内存向量库，形成可检索索引。"""

import threading
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from agent.knowledge_embeddings import create_knowledge_embeddings
from agent.knowledge_loader import load_knowledge_documents
from agent.knowledge_splitter import split_knowledge_documents


def build_knowledge_index(
    chunks: list[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    """用外部传入的 Embeddings 为片段建立内存向量索引。

    从每个片段的 ``metadata["chunk_id"]`` 依次收集 ID，与 ``chunks`` 的
    顺序一一对应；缺失或重复的 ID 直接抛出 ``ValueError``，避免向量库中
    后写入的片段覆盖先写入的同名片段。校验通过后创建
    ``InMemoryVectorStore``，把完整 ``Document``（正文、标题、来源等元数据）
    与 ID 一起传给 ``add_documents``，再返回索引供查询使用。
    Embeddings 由调用方传入，便于复用已经加载的模型。
    """
    ids: list[str] = []
    seen_ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError(f"第 {index} 个片段缺少 chunk_id，无法建立索引")
        chunk_id = str(chunk_id)
        if chunk_id in seen_ids:
            raise ValueError(
                f"第 {index} 个片段的 chunk_id 重复：{chunk_id}，无法建立索引"
            )
        seen_ids.add(chunk_id)
        ids.append(chunk_id)

    index = InMemoryVectorStore(embeddings)
    index.add_documents(documents=chunks, ids=ids)
    return index


def create_knowledge_index_provider(
    knowledge_dir: Path,
) -> Callable[[], InMemoryVectorStore]:
    """创建按需建库、可并发安全复用的索引提供器。

    外层函数只保存知识库目录，并为提供器准备一把 ``threading.Lock`` 和一个
    初始为 ``None`` 的索引变量；此时既不读取文件，也不加载 Embedding 模型。
    返回的 ``get_index`` 首次调用时在锁内检查索引是否已存在，若不存在则按
    ``load_knowledge_documents → split_knowledge_documents →
    create_knowledge_embeddings → build_knowledge_index`` 的顺序建库，成功后
    保存并返回。锁同时覆盖“检查”和“首次建库”，避免 LangGraph 并发执行同一轮
    多个工具请求时重复建库。建库失败时原异常直接抛出，失败结果不会被保存，
    因此下一次调用仍可重新尝试。
    """

    index: InMemoryVectorStore | None = None
    lock = threading.Lock()

    def get_index() -> InMemoryVectorStore:
        nonlocal index
        with lock:
            if index is None:
                documents = load_knowledge_documents(knowledge_dir)
                chunks = split_knowledge_documents(documents)
                embeddings = create_knowledge_embeddings()
                index = build_knowledge_index(chunks, embeddings)
            return index

    return get_index
