"""知识库切分层：按标题与长度把整篇文档拆成可检索片段。"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def split_knowledge_documents(
    documents: list[Document],
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Document]:
    """把加载器输出的整篇文档切分为带完整元数据的片段。

    先按 Markdown 标题拆成小节，``strip_headers=False`` 让标题保留在正文中；
    每个小节合并原文档的元数据，继续携带 ``source_id``。再用递归字符切分器
    按长度拆分过长小节，``chunk_size`` 与 ``chunk_overlap`` 以字符数计量。
    输出片段先复制小节继承的全部元数据，再补齐 ``document_title``、
    ``section_title`` 与 ``chunk_id``，``chunk_id`` 在同一源文件内从 1
    连续编号，处理下一个文件时重新开始。
    文件顺序与文件内部片段顺序保持不变，原始 ``Document`` 不被修改。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须大于 0，当前为：{chunk_size}")
    if chunk_overlap < 0 or chunk_overlap > chunk_size:
        raise ValueError(
            "chunk_overlap 必须满足 0 ≤ chunk_overlap ≤ chunk_size，"
            f"当前为：{chunk_overlap}，chunk_size 为：{chunk_size}"
        )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "document_title"),
            ("##", "section_title"),
        ],
        strip_headers=False,
    )
    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            "，",
            ",",
            " ",
            "",
        ],
    )

    chunks: list[Document] = []
    for document in documents:
        sections = header_splitter.split_text(document.page_content)
        for section in sections:
            section.metadata = {**document.metadata, **section.metadata}

        split_chunks = length_splitter.split_documents(sections)
        source_id = str(document.metadata.get("source_id", ""))
        for index, chunk in enumerate(split_chunks, start=1):
            metadata = dict(chunk.metadata)
            metadata["source_id"] = source_id
            metadata.setdefault("document_title", "")
            metadata.setdefault("section_title", "")
            metadata["chunk_id"] = f"{source_id}#chunk-{index:04d}"
            chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )
    return chunks
