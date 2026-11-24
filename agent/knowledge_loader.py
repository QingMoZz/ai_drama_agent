"""知识库加载层：把磁盘上的 Markdown 文件转成 LangChain Document。"""

from pathlib import Path

from langchain_core.documents import Document


def load_knowledge_documents(knowledge_dir: Path) -> list[Document]:
    """读取知识库目录直属的 Markdown 文件并转成 Document 列表。

    只处理目录第一层的 ``.md`` 文件，按文件名排序后逐个以 UTF-8 读取；
    ``page_content`` 保存完整 Markdown 原文，``metadata.source_id`` 记录相对
    知识库根目录的路径（例如 ``clue_reveal.md``），便于后续片段返回时核对
    出处。目录不存在或文件读取失败时，异常直接抛给调用方。
    """
    paths = [
        path
        for path in knowledge_dir.iterdir()
        if path.is_file() and path.suffix == ".md"
    ]

    documents: list[Document] = []
    for path in sorted(paths, key=lambda item: item.name):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source_id": path.relative_to(knowledge_dir).as_posix()},
            )
        )
    return documents
