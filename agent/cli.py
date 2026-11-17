"""命令行接口：定义要执行的命令及其参数。"""

import argparse
from pathlib import Path


def build_parser(default_db_path: Path) -> argparse.ArgumentParser:
    """构建命令行解析器，定义 ``generate`` 与 ``show`` 两个子命令。

    解析结果中的 ``command`` 保存子命令名称；``--episode-id`` 对应
    ``episode_id``，``--revision`` 省略时为 ``None``，表示最新版本。
    ``--db-path`` 的默认值由调用方通过 ``default_db_path`` 传入。
    """
    parser = argparse.ArgumentParser(
        description="AI 漫剧生成助手：生成并查看整集创作快照。"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="要执行的子命令",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="生成一集内容并保存为新版本",
        description="按创作要求生成梗概、剧本与分镜，并保存为该集的新版本。",
    )
    generate_parser.add_argument(
        "--episode-id",
        required=True,
        help="本次生成内容所属的集 ID；同一集继续生成新版本时保持不变",
    )
    generate_parser.add_argument(
        "--requirement",
        required=True,
        help="用户的创作要求，保留输入原文",
    )
    generate_parser.add_argument(
        "--db-path",
        type=Path,
        default=default_db_path,
        help="SQLite 数据库文件路径；省略时使用项目默认数据库",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="查看某一集的最新或指定版本快照",
        description="读取某一集的完整快照；不指定版本时展示最新版本。",
    )
    show_parser.add_argument(
        "--episode-id",
        required=True,
        help="要查看的集 ID",
    )
    show_parser.add_argument(
        "--revision",
        type=int,
        default=None,
        help="指定历史版本号（大于等于 1）；省略时表示最新版本",
    )
    show_parser.add_argument(
        "--db-path",
        type=Path,
        default=default_db_path,
        help="SQLite 数据库文件路径；省略时使用项目默认数据库",
    )

    return parser
