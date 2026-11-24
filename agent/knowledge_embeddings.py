"""知识库向量化层：提供基于本地 BGE 中文模型的 Embeddings 实现。"""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

BGE_QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："


def create_knowledge_embeddings() -> Embeddings:
    """创建本地 BGE 中文小模型的 Embeddings 实例。

    模型在函数调用时初始化并加载到 CPU，首次运行会从 HuggingFace 下载
    ``BAAI/bge-small-zh-v1.5``。文档与查询都开启 ``normalize_embeddings``，
    使向量长度接近 1；查询另外通过 ``query_encode_kwargs`` 加上 BGE 推荐的
    检索前缀，因此归一化需要在这组配置中显式重复。
    """
    return HuggingFaceEmbeddings(
        model="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": BGE_QUERY_PROMPT,
        },
    )
