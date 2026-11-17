"""存储层：负责数据库初始化与快照读写。"""

import sqlite3
from contextlib import closing
from pathlib import Path

from agent.snapshot_schema import EpisodeSnapshot, EpisodeSnapshotPayload


def initialize_database(db_path: Path) -> None:
    """初始化数据库：创建 ``episode_snapshots`` 快照表。

    数据库文件位置由调用方通过 ``db_path`` 指定。建表使用
    ``IF NOT EXISTS``，对同一个数据库重复调用不会影响已有表和数据。
    成功时返回 ``None``；发生数据库错误时向调用方抛出异常。
    连接对象的上下文管理器负责提交或回滚，关闭动作由 ``closing`` 负责。
    """
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_snapshots (
                    episode_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (episode_id, revision)
                )
                """
            )


def save_episode_snapshot(
    db_path: Path,
    episode_id: str,
    payload: EpisodeSnapshotPayload,
) -> int:
    """把一集的完整内容保存为一个新版本，提交成功后返回版本号。

    写入前先导出为普通字典并重新校验，防止模型创建后内部列表被修改导致
    约束失效；这一步在开启写事务之前完成，缩短写事务占用时间。事务内先执行
    ``BEGIN IMMEDIATE``，再在同一连接中查询该集最大版本号并插入新记录，
    旧版本保持不变。失败时异常抛给调用方，由事务回滚、连接关闭。
    """
    validated = EpisodeSnapshotPayload.model_validate(payload.model_dump())
    payload_json = validated.model_dump_json()

    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")

            row = connection.execute(
                "SELECT MAX(revision) FROM episode_snapshots WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            revision = 1 if row[0] is None else row[0] + 1

            connection.execute(
                "INSERT INTO episode_snapshots (episode_id, revision, payload_json) "
                "VALUES (?, ?, ?)",
                (episode_id, revision, payload_json),
            )

    return revision


def load_episode_snapshot(
    db_path: Path,
    episode_id: str,
    revision: int | None = None,
) -> EpisodeSnapshot | None:
    """读取某集的最新版本或指定版本，返回完整快照或 ``None``。

    ``revision`` 为 ``None`` 时按版本号降序取第一条；指定版本时按
    ``episode_id`` 与 ``revision`` 精确查询。没有匹配记录（集不存在或版本
    不存在）时返回 ``None``。连接关闭后再解析 ``payload_json``，由
    ``EpisodeSnapshotPayload`` 重新执行字段与内容校验；数据库错误、JSON
    损坏或校验失败时异常继续向上抛出。
    """
    if revision is not None and revision < 1:
        raise ValueError(f"revision 必须大于等于 1，当前为：{revision}")

    with closing(sqlite3.connect(db_path)) as connection:
        if revision is None:
            row = connection.execute(
                "SELECT episode_id, revision, payload_json, created_at "
                "FROM episode_snapshots WHERE episode_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT episode_id, revision, payload_json, created_at "
                "FROM episode_snapshots WHERE episode_id = ? AND revision = ?",
                (episode_id, revision),
            ).fetchone()

    if row is None:
        return None

    payload = EpisodeSnapshotPayload.model_validate_json(row[2])
    return EpisodeSnapshot(
        episode_id=row[0],
        revision=row[1],
        created_at=row[3],
        payload=payload,
    )
