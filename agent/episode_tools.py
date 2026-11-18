"""工具层：向模型暴露已保存剧集的查询能力。"""

from dataclasses import dataclass
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from agent.storage import load_episode_snapshot


@dataclass(frozen=True)
class EpisodeToolContext:
    """工具运行上下文：承载运行配置，数据库访问发生在工具执行时。"""

    db_path: Path


class EpisodeSummaryInput(BaseModel):
    """工具输入契约：模型只能看到集 ID 与可选的版本号。"""

    episode_id: str = Field(description="要查询的集 ID")
    revision: int | None = Field(
        default=None,
        ge=1,
        description="要读取的版本号（大于等于 1）；省略时读取最新版本",
    )


@tool(args_schema=EpisodeSummaryInput)
def get_episode_summary(
    episode_id: str,
    runtime: ToolRuntime[EpisodeToolContext],
    revision: int | None = None,
) -> dict[str, object] | None:
    """查询已保存剧集的梗概、角色与统计信息。

    用于了解某一集保存过什么内容：返回剧本标题、故事梗概、角色清单
    （仅 character_id 与 name）、按播放顺序排列的场号、场景数、镜头总数
    以及各场预计时长之和。指定 revision 时读取该历史版本，省略时读取
    最新版本，返回中的 revision 始终是数据库中的实际版本号。

    没有匹配的记录时返回 None；数据库文件不存在、读取失败、JSON 损坏或
    内容校验不通过时抛出异常。
    """
    db_path = runtime.context.db_path
    if not db_path.exists():
        raise FileNotFoundError(f"数据库文件不存在：{db_path}")

    snapshot = load_episode_snapshot(db_path, episode_id, revision)
    if snapshot is None:
        return None

    script = snapshot.payload.script
    storyboard = snapshot.payload.storyboard

    return {
        "episode_id": snapshot.episode_id,
        "revision": snapshot.revision,
        "created_at": snapshot.created_at,
        "title": script.title,
        "story": snapshot.payload.story,
        "characters": [
            {"character_id": character.character_id, "name": character.name}
            for character in script.characters
        ],
        "scene_numbers": [scene.scene_no for scene in script.scenes],
        "scene_count": len(script.scenes),
        "shot_count": sum(len(scene.shots) for scene in storyboard.scenes),
        "total_duration_sec": sum(scene.duration_sec for scene in script.scenes),
    }
