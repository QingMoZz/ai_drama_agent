"""分镜的数据契约：定义镜头、单场分镜、整集分镜，以及与剧本的核对。"""

from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, StringConstraints, model_validator

from agent.script_schema import EpisodeScript


class StoryboardShot(BaseModel):
    """分镜镜头：记录镜头 ID、画面可见内容与出镜角色。"""

    shot_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(description="镜头 ID，清理首尾空白后不能为空")
    visual_description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(description="观众在画面里能看到的内容，清理首尾空白后不能为空")
    character_ids: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(
        default_factory=list,
        description="实际出现在画面中的角色 ID 列表，空镜头允许为空列表",
    )


class SceneStoryboard(BaseModel):
    """单场分镜：场号与按播放顺序排列的镜头列表。"""

    scene_no: int = Field(description="场号，沿用剧本场景的场号")
    shots: list[StoryboardShot] = Field(
        min_length=1, description="按播放顺序排列的镜头，至少包含一个镜头"
    )

    @model_validator(mode="after")
    def _validate_shot_ids_unique(self) -> "SceneStoryboard":
        """检查同一场内镜头 ID 不重复。

        单个镜头看不到同场的其他镜头，因此重复检查放在能读取整个
        ``shots`` 列表的场级模型上；跨场景的重复留给整集层校验。
        """
        seen: set[str] = set()
        for shot in self.shots:
            if shot.shot_id in seen:
                raise ValueError(
                    f"第 {self.scene_no} 场镜头 ID 重复：{shot.shot_id}"
                )
            seen.add(shot.shot_id)
        return self


class EpisodeStoryboard(BaseModel):
    """整集分镜：按播放顺序排列的场级分镜。"""

    scenes: list[SceneStoryboard] = Field(
        min_length=1, description="按播放顺序排列的场级分镜，至少包含一场"
    )

    @model_validator(mode="after")
    def _validate_storyboard_integrity(self) -> "EpisodeStoryboard":
        """在整集层检查场号唯一与镜头 ID 本集内唯一。

        遍历各场镜头时记录每个 ``shot_id`` 第一次出现的场号，重复时
        同时给出镜头 ID 与两个场号，便于定位。
        """
        scene_nos: set[int] = set()
        first_scene_by_shot: dict[str, int] = {}
        for scene in self.scenes:
            if scene.scene_no in scene_nos:
                raise ValueError(f"场号重复：{scene.scene_no}")
            scene_nos.add(scene.scene_no)

            for shot in scene.shots:
                first_scene_no = first_scene_by_shot.get(shot.shot_id)
                if first_scene_no is not None:
                    raise ValueError(
                        f"镜头 ID 重复：{shot.shot_id}，"
                        f"分别出现在第 {first_scene_no} 场与第 {scene.scene_no} 场"
                    )
                first_scene_by_shot[shot.shot_id] = scene.scene_no
        return self


class StoryboardShotDraft(BaseModel):
    """分镜镜头草稿：模型生成阶段还没有 ``shot_id``。"""

    visual_description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(description="观众在画面里能看到的内容，清理首尾空白后不能为空")
    character_ids: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(
        default_factory=list,
        description="实际出现在画面中的角色 ID 列表，空镜头允许为空列表",
    )


class SceneStoryboardDraft(BaseModel):
    """单场分镜草稿：场号与按播放顺序排列的镜头草稿列表。"""

    scene_no: int = Field(description="场号，沿用剧本场景的场号")
    shots: list[StoryboardShotDraft] = Field(
        min_length=1, description="按播放顺序排列的镜头草稿，至少包含一个镜头"
    )


class EpisodeStoryboardDraft(BaseModel):
    """整集分镜草稿：保留输入场次顺序的场级草稿列表。"""

    scenes: list[SceneStoryboardDraft] = Field(
        min_length=1, description="按场次顺序排列的场级分镜草稿，至少包含一场"
    )


def validate_storyboard_against_script(
    script: EpisodeScript,
    storyboard: EpisodeStoryboard,
) -> None:
    """核对整集分镜与原始剧本：场号覆盖范围、播放顺序与出镜角色登记。

    分镜自身的结构完整性由模型校验保证；这里补充只有对照剧本才能发现的
    问题：场号缺失或多出、场级播放顺序与剧本列表不一致，以及镜头引用了
    剧本角色表中未登记的角色 ID。
    成功时正常返回，失败时抛出带具体原因的 ``ValueError``。
    """
    script_scene_nos = {scene.scene_no for scene in script.scenes}
    storyboard_scene_nos = {scene.scene_no for scene in storyboard.scenes}

    missing = sorted(script_scene_nos - storyboard_scene_nos)
    extra = sorted(storyboard_scene_nos - script_scene_nos)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("缺第 " + "、".join(str(no) for no in missing) + " 场")
        if extra:
            details.append("多第 " + "、".join(str(no) for no in extra) + " 场")
        raise ValueError("分镜场次与剧本不一致：" + "，".join(details))

    # 场次数量与场号唯一性已由前面的检查保证，逐位置比较即可判断播放顺序；
    # 以剧本列表顺序为准，即使场号不连续或不递增。
    expected_order = [scene.scene_no for scene in script.scenes]
    actual_order = [scene.scene_no for scene in storyboard.scenes]
    for position, (expected_no, actual_no) in enumerate(
        zip(expected_order, actual_order), start=1
    ):
        if expected_no != actual_no:
            raise ValueError(
                f"分镜播放顺序不一致，第 {position} 个位置应为第 "
                f"{expected_no} 场，实际为第 {actual_no} 场"
            )

    registered_ids = {
        character.character_id for character in script.characters
    }
    for scene in storyboard.scenes:
        for shot in scene.shots:
            for character_id in shot.character_ids:
                if character_id not in registered_ids:
                    raise ValueError(
                        f"第 {scene.scene_no} 场镜头 {shot.shot_id} "
                        f"引用了未登记的角色 ID：{character_id}"
                    )


def finalize_storyboard(
    script: EpisodeScript,
    draft: EpisodeStoryboardDraft,
) -> EpisodeStoryboard:
    """把分镜草稿转换为正式分镜，并通过全部一致性检查。

    按草稿的场次与镜头顺序遍历，为每个镜头单独生成 UUID4 字符串作为
    ``shot_id``，再组装正式模型，由 ``EpisodeStoryboard`` 检查场号与镜头 ID
    唯一性；最后对照原始剧本核对场次覆盖、播放顺序与角色引用。
    全部通过后返回正式分镜。
    """
    scenes = [
        SceneStoryboard(
            scene_no=scene_draft.scene_no,
            shots=[
                StoryboardShot(
                    shot_id=str(uuid4()),
                    visual_description=shot_draft.visual_description,
                    character_ids=shot_draft.character_ids,
                )
                for shot_draft in scene_draft.shots
            ],
        )
        for scene_draft in draft.scenes
    ]
    storyboard = EpisodeStoryboard(scenes=scenes)
    validate_storyboard_against_script(script, storyboard)
    return storyboard
