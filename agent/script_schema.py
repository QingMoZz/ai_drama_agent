"""单集剧本的数据契约：定义角色、对白、场景与单集结构。"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


class CharacterAlias(BaseModel):
    """角色称呼：记录称呼文本、使用范围与可使用该称呼的角色。"""

    alias: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(description="称呼文本，清理首尾空白后不能为空")
    scope: Literal["public", "restricted"] = Field(
        description="使用范围：public 为任何角色可用，restricted 为仅限指定角色"
    )
    used_by_character_ids: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(description="使用者角色 ID 列表，public 时必须为空")
    note: str | None = Field(default=None, description="使用场合、语气等说明")

    @model_validator(mode="after")
    def _validate_scope_usage(self) -> "CharacterAlias":
        """检查使用范围与使用者列表是否一致。"""
        if self.scope == "public" and self.used_by_character_ids:
            raise ValueError(
                "public（公共）称呼不应填写限定使用者，"
                "used_by_character_ids 必须为空，"
                f"当前为：{self.used_by_character_ids}"
            )
        if self.scope == "restricted" and not self.used_by_character_ids:
            raise ValueError(
                "restricted（限定）称呼必须至少填写一个非空使用者角色 ID，"
                "used_by_character_ids 不能为空"
            )
        return self


class ScriptCharacter(BaseModel):
    """剧本角色：记录角色 ID、显示名称与称呼别名。"""

    character_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(description="角色 ID")
    name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(description="显示名称")
    aliases: list[CharacterAlias] = Field(
        default_factory=list, description="该角色的称呼别名"
    )


class DialogueLine(BaseModel):
    """一句对白：记录说话人角色 ID 与台词文本。"""

    character_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(description="说话人角色 ID")
    line: str = Field(min_length=1, description="台词文本")


class ScriptScene(BaseModel):
    """剧本场景：场号、地点、预计时长、动作与对白。"""

    scene_no: int = Field(description="场号")
    location: str = Field(min_length=1, description="地点")
    duration_sec: int = Field(gt=0, description="预计时长（正整数秒）")
    action: str = Field(default="", description="动作描述")
    dialogue: list[DialogueLine] = Field(default_factory=list, description="对白")


class EpisodeScript(BaseModel):
    """单集剧本：标题、角色清单与按剧情顺序排列的场景。"""

    title: str = Field(min_length=1, description="标题")
    characters: list[ScriptCharacter] = Field(min_length=1, description="角色清单")
    scenes: list[ScriptScene] = Field(min_length=1, description="场景列表")

    @model_validator(mode="after")
    def _validate_episode_integrity(self) -> "EpisodeScript":
        """在整集层检查跨场景的完整性约束。"""
        scene_nos: set[int] = set()
        for scene in self.scenes:
            if scene.scene_no in scene_nos:
                raise ValueError(f"场号重复：{scene.scene_no}")
            scene_nos.add(scene.scene_no)

        registered_ids: set[str] = set()
        for character in self.characters:
            if character.character_id in registered_ids:
                raise ValueError(f"角色 ID 重复：{character.character_id}")
            registered_ids.add(character.character_id)

        for scene in self.scenes:
            for line in scene.dialogue:
                if line.character_id not in registered_ids:
                    raise ValueError(
                        f"第 {scene.scene_no} 场引用了未登记的角色 ID："
                        f"{line.character_id}"
                    )

        for character in self.characters:
            for alias in character.aliases:
                for user_id in alias.used_by_character_ids:
                    if user_id not in registered_ids:
                        raise ValueError(
                            f"角色 {character.character_id} 的称呼"
                            f"「{alias.alias}」引用了未登记的使用者角色 ID："
                            f"{user_id}"
                        )
        return self
