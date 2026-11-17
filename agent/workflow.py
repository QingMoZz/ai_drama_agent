"""工作流层：负责定义状态、节点和执行流程。"""

from typing import NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, UsageMetadata
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.script_schema import EpisodeScript
from agent.storyboard_schema import (
    EpisodeStoryboard,
    EpisodeStoryboardDraft,
    finalize_storyboard,
)

# 各节点每次调用的生成预算（输出 Token 上限），实际用量仍以模型返回的用量为准。
STORY_MAX_TOKENS = 1024
SCRIPT_MAX_TOKENS = 4096
STORYBOARD_MAX_TOKENS = 4096

SYSTEM_PROMPT = "你是一名擅长创作悬疑故事的漫剧编剧，你需要使用中文交流。"

SCRIPT_SYSTEM_PROMPT = (
    "你是一名擅长创作悬疑故事的漫剧编剧，负责把故事梗概扩写成单集剧本。"
    "必须使用中文，并且只输出一个 JSON 对象，不要输出解释文字或代码块标记。"
    "产物必须包含角色清单 characters、按剧情顺序排列的场景 scenes，"
    "以及每个场景中的对白 dialogue。"
    "对白和称呼的使用者都必须通过 character_id 引用角色清单中已登记的角色。"
    "JSON 结构："
    '{"title": 标题, '
    '"characters": [{"character_id": 角色 ID, "name": 显示名称, '
    '"aliases": [{"alias": 称呼, "scope": "public" 或 "restricted", '
    '"used_by_character_ids": [角色 ID], "note": 说明或 null}]}], '
    '"scenes": [{"scene_no": 场号, "location": 地点, '
    '"duration_sec": 正整数秒, "action": 动作, '
    '"dialogue": [{"character_id": 角色 ID, "line": 台词}]}]}'
)

STORYBOARD_SYSTEM_PROMPT = (
    "你是一名擅长创作悬疑故事的漫剧分镜师，负责把单集剧本拆解成可直接拍摄的分镜。"
    "必须使用中文，把每场剧本转化为具体、可见的画面，只描述观众能在画面里看到的内容。"
    "必须完整覆盖剧本中的全部场景，并保持原剧本的场次顺序；"
    "每场至少生成一个镜头，镜头列表按播放顺序排列。"
    "出镜角色必须通过 character_id 引用剧本角色表中已登记的角色；"
    "没有角色出镜的空镜头，character_ids 使用空列表。"
    "按以下 JSON 结构提供场号、画面描述与出镜角色："
    '{"scenes": [{"scene_no": 场号, '
    '"shots": [{"visual_description": 画面描述, '
    '"character_ids": [角色 ID]}]}]}'
)


class WorkflowState(TypedDict):
    """故事生成工作流的状态。

    ``requirement`` 是启动时必须提供的创作要求；``story``、``script``、
    ``storyboard`` 与 ``usage_by_node`` 在流程推进过程中才逐步产生，因此标记为
    ``NotRequired``，允许在生成之前不存在。``storyboard`` 是已完成 UUID 分配、
    且通过剧本核对的整集分镜。``usage_by_node`` 记录各节点的模型调用用量，
    以节点名称为键，例如 ``generate_story``、``generate_script`` 与
    ``generate_storyboard``；服务端未提供某个节点的用量时，该节点的记录保持缺失。
    """

    requirement: str
    story: NotRequired[str]
    script: NotRequired[EpisodeScript]
    storyboard: NotRequired[EpisodeStoryboard]
    usage_by_node: NotRequired[dict[str, UsageMetadata]]


def generate_story(
    state: WorkflowState,
    model: BaseChatModel,
) -> dict[str, object]:
    """生成节点：调用模型，把创作要求转换成故事文本。

    以 ``SYSTEM_PROMPT`` 为角色设定，以状态中的 ``requirement`` 作为用户消息；
    成功时返回待合并进状态的 ``story``。服务端提供了用量时，复制状态中已有的
    ``usage_by_node``（不存在则按空字典处理），在新字典中登记 ``generate_story``
    的用量后一并返回；没有用量则省略该字段更新。
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["requirement"]),
    ]

    response = model.invoke(messages, max_tokens=STORY_MAX_TOKENS)

    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError("输出被截断，请调整生成长度或创作要求。")
    if finish_reason != "stop":
        raise RuntimeError(f"模型没有正常完成生成：{finish_reason}")

    story = response.text.strip()
    if not story:
        raise RuntimeError("模型返回了空文本。")

    update: dict[str, object] = {"story": story}
    if response.usage_metadata is not None:
        usage_by_node = dict(state.get("usage_by_node", {}))
        usage_by_node["generate_story"] = response.usage_metadata
        update["usage_by_node"] = usage_by_node
    return update


def generate_script(
    state: WorkflowState,
    model: BaseChatModel,
) -> dict[str, object]:
    """编剧节点：把原始要求与故事梗概扩写成经过校验的单集剧本。

    请求同时携带 ``requirement`` 与 ``story``；模型以 ``function_calling``
    的方式按 ``EpisodeScript`` 约束返回结构化结果。原始响应用于检查结束
    原因与读取用量，解析结果与解析错误分别用于取剧本和报错。检查全部通过
    后返回 ``script``；``raw`` 提供了用量时，复制状态中已有的 ``usage_by_node``
    （不存在则按空字典处理），在新字典中登记 ``generate_script`` 的用量后一并返回。
    """
    story = state.get("story")
    if not story:
        raise ValueError("状态缺少 story，请先执行梗概生成节点。")

    messages = [
        SystemMessage(content=SCRIPT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"原始创作要求：\n{state['requirement']}\n\n"
                f"故事梗概：\n{story}\n\n"
                "请依据以上要求与梗概，输出单集剧本的 JSON。"
            )
        ),
    ]

    structured_model = model.with_structured_output(
        EpisodeScript,
        method="function_calling",
        include_raw=True,
        max_tokens=SCRIPT_MAX_TOKENS,
    )
    result = structured_model.invoke(messages)

    raw = result["raw"]
    parsed = result["parsed"]
    parsing_error = result["parsing_error"]

    finish_reason = raw.response_metadata.get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError("剧本输出被截断，请缩短要求或提高生成长度上限。")
    if finish_reason not in ("stop", "tool_calls"):
        raise RuntimeError(f"模型没有正常完成生成：{finish_reason}")

    if parsing_error is not None:
        raise RuntimeError(f"剧本未通过数据契约校验：{parsing_error}") from parsing_error

    if parsed is None:
        raise RuntimeError("模型没有返回可解析的剧本。")

    update: dict[str, object] = {"script": parsed}
    if raw.usage_metadata is not None:
        usage_by_node = dict(state.get("usage_by_node", {}))
        usage_by_node["generate_script"] = raw.usage_metadata
        update["usage_by_node"] = usage_by_node
    return update


def generate_storyboard(
    state: WorkflowState,
    model: BaseChatModel,
) -> dict[str, object]:
    """分镜节点：把已校验的剧本拆解成经过校验的整集分镜。

    请求同时携带 ``requirement`` 与完整剧本的 JSON 文本。模型以
    ``function_calling`` 的方式按 ``EpisodeStoryboardDraft`` 返回结构化草稿，
    草稿不含 ``shot_id``；结束原因、解析错误与解析结果检查通过后，交由
    ``finalize_storyboard`` 分配 UUID 并核对场次覆盖、播放顺序与角色引用。
    成功时返回正式 ``storyboard``；``raw`` 提供了用量时，复制状态中已有的
    ``usage_by_node``（不存在则按空字典处理），登记 ``generate_storyboard``
    的用量后一并返回。
    """
    script = state.get("script")
    if script is None:
        raise ValueError("状态缺少 script，请先执行编剧节点。")

    messages = [
        SystemMessage(content=STORYBOARD_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"原始创作要求：\n{state['requirement']}\n\n"
                f"单集剧本：\n{script.model_dump_json(indent=2)}\n\n"
                "请依据以上要求与剧本，输出整集分镜的 JSON。"
            )
        ),
    ]

    structured_model = model.with_structured_output(
        EpisodeStoryboardDraft,
        method="function_calling",
        include_raw=True,
        max_tokens=STORYBOARD_MAX_TOKENS,
    )
    result = structured_model.invoke(messages)

    raw = result["raw"]
    parsed = result["parsed"]
    parsing_error = result["parsing_error"]

    finish_reason = raw.response_metadata.get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError("分镜输出被截断，请缩短要求或提高生成长度上限。")
    if finish_reason not in ("stop", "tool_calls"):
        raise RuntimeError(f"模型没有正常完成生成：{finish_reason}")

    if parsing_error is not None:
        raise RuntimeError(f"分镜未通过数据契约校验：{parsing_error}") from parsing_error

    if parsed is None:
        raise RuntimeError("模型没有返回可解析的分镜草稿。")

    storyboard = finalize_storyboard(script, parsed)

    update: dict[str, object] = {"storyboard": storyboard}
    if raw.usage_metadata is not None:
        usage_by_node = dict(state.get("usage_by_node", {}))
        usage_by_node["generate_storyboard"] = raw.usage_metadata
        update["usage_by_node"] = usage_by_node
    return update


def build_workflow(model: BaseChatModel) -> CompiledStateGraph:
    """构建并编译故事生成工作流。

    构图时把模型客户端固定下来：注册到图中的节点只接收 LangGraph 传入的
    状态，再连同模型一起转交给对应的节点逻辑。执行顺序为
    ``START → generate_story → generate_script → generate_storyboard → END``，
    正常完成时三个节点各调用一次模型，共三次；前一个节点失败时，后一个节点
    不会执行。调用方拿到编译后的图，传入包含 ``requirement`` 的初始状态即可
    执行一次任务。
    """

    def generate_story_node(state: WorkflowState) -> dict[str, object]:
        """图中的生成节点：把状态与已绑定的模型转交给生成逻辑。"""
        return generate_story(state, model)

    def generate_script_node(state: WorkflowState) -> dict[str, object]:
        """图中的编剧节点：把状态与已绑定的模型转交给编剧逻辑。"""
        return generate_script(state, model)

    def generate_storyboard_node(state: WorkflowState) -> dict[str, object]:
        """图中的分镜节点：把状态与已绑定的模型转交给分镜逻辑。"""
        return generate_storyboard(state, model)

    builder = StateGraph(WorkflowState)

    builder.add_node("generate_story", generate_story_node)
    builder.add_node("generate_script", generate_script_node)
    builder.add_node("generate_storyboard", generate_storyboard_node)

    builder.add_edge(START, "generate_story")
    builder.add_edge("generate_story", "generate_script")
    builder.add_edge("generate_script", "generate_storyboard")
    builder.add_edge("generate_storyboard", END)

    graph = builder.compile()
    return graph
