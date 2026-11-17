"""快照内容的数据契约：定义 episode_snapshots.payload_json 保存的整集内容。"""

from langchain_core.messages import UsageMetadata
from pydantic import BaseModel, Field, model_validator

from agent.script_schema import EpisodeScript
from agent.storyboard_schema import (
    EpisodeStoryboard,
    validate_storyboard_against_script,
)


class EpisodeSnapshotPayload(BaseModel):
    """一集某个版本的完整内容。

    ``episode_id``、``revision`` 与 ``created_at`` 由数据库行负责标识版本，
    本模型只描述该版本的内容。``usage_by_node`` 记录各节点的模型调用用量，
    未提供时使用独立的空字典。
    """

    requirement: str = Field(description="原始创作要求")
    story: str = Field(description="故事梗概")
    script: EpisodeScript = Field(description="单集剧本")
    storyboard: EpisodeStoryboard = Field(description="正式整集分镜")
    usage_by_node: dict[str, UsageMetadata] = Field(
        default_factory=dict,
        description="节点名称到模型调用用量的映射",
    )

    @model_validator(mode="after")
    def _validate_script_storyboard_match(self) -> "EpisodeSnapshotPayload":
        """检查快照中的剧本与分镜是否匹配。

        复用与生成流程相同的核对规则：场次覆盖、播放顺序与角色引用。
        """
        validate_storyboard_against_script(self.script, self.storyboard)
        return self


class EpisodeSnapshot(BaseModel):
    """某一集某一个版本的完整快照：外层管理版本信息，内层管理创作内容。"""

    episode_id: str = Field(description="这份快照所属的集 ID")
    revision: int = Field(ge=1, description="内容版本号，从 1 开始")
    created_at: str = Field(
        description="数据库保存的创建时间，UTC，格式为 YYYY-MM-DD HH:MM:SS"
    )
    payload: EpisodeSnapshotPayload = Field(
        description="已解析、经过校验的完整创作内容"
    )
