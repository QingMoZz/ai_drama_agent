"""工作流层：负责定义状态、节点和执行流程。"""

from collections.abc import Callable
from typing import NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, UsageMetadata
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.knowledge_retrieval import KnowledgeHit, retrieve_knowledge
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

SYSTEM_PROMPT = (
    "你是一名擅长创作悬疑故事的漫剧编剧，你需要使用中文交流。"
    "你的任务是写一段约 120–180 字的中文故事梗概，直接输出故事正文，"
    "不要标题、前缀说明或解释文字。"
    "梗概要交代主角、触发事件、悬念如何发展，以及结尾揭示了什么；"
    "场次、对白和镜头细节留给后续节点，不要在梗概中展开。"
    "严格遵守用户的创作要求：例如要求主角先捡到车票时，可以写他捡到票，"
    "但要把日期异常安排在结尾才揭示。"
)

SCRIPT_SYSTEM_PROMPT = (
    "你是一名擅长创作悬疑故事的漫剧编剧，负责把故事梗概扩写成单集剧本。"
    "必须使用中文，并且只输出一个 JSON 对象，不要输出解释文字或代码块标记。"
    "产物必须包含角色清单 characters、按剧情顺序排列的场景 scenes，"
    "以及每个场景中的对白 dialogue。"
    "对白和称呼的使用者都必须通过 character_id 引用角色清单中已登记的角色。"
    "若提供了知识库参考，可参考其中的悬念节奏与线索铺垫，"
    "但原始创作需求、故事梗概与结构化输出约束优先，仍按原结构返回。"
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
    "若提供了知识库参考，可参考其中的构图方式与揭示时机，"
    "但原始创作需求、单集剧本与输出结构约束优先，仍按原结构返回。"
    "按以下 JSON 结构提供场号、画面描述与出镜角色："
    '{"scenes": [{"scene_no": 场号, '
    '"shots": [{"visual_description": 画面描述, '
    '"character_ids": [角色 ID]}]}]}'
)


class WorkflowState(TypedDict):
    """故事生成工作流的状态。

    ``requirement`` 是启动时必须提供的创作要求；``story``、``script``、
    ``storyboard``、``knowledge_hits`` 与 ``usage_by_node`` 在流程推进过程中才
    逐步产生，因此标记为 ``NotRequired``，允许在生成之前不存在。
    ``knowledge_hits`` 由检索节点依据原始需求产生，供后续生成节点复用。
    ``storyboard`` 是已完成 UUID 分配、且通过剧本核对的整集分镜。
    ``usage_by_node`` 记录各节点的模型调用用量，
    以节点名称为键，例如 ``generate_story``、``generate_script`` 与
    ``generate_storyboard``；服务端未提供某个节点的用量时，该节点的记录保持缺失。
    """

    requirement: str
    story: NotRequired[str]
    script: NotRequired[EpisodeScript]
    storyboard: NotRequired[EpisodeStoryboard]
    knowledge_hits: NotRequired[list[KnowledgeHit]]
    usage_by_node: NotRequired[dict[str, UsageMetadata]]


def retrieve_knowledge_context(
    state: WorkflowState,
    knowledge_index_provider: Callable[[], InMemoryVectorStore],
) -> dict[str, object]:
    """检索节点：用原始创作需求查询知识库，产生 ``knowledge_hits``。

    从状态读取 ``requirement`` 作为查询，调用索引提供器取得索引后交给
    ``retrieve_knowledge``；返回只包含 ``knowledge_hits`` 的状态更新，由
    LangGraph 合并。空索引时照样写入，值为空列表。
    """
    index = knowledge_index_provider()
    hits = retrieve_knowledge(index, state["requirement"])
    return {"knowledge_hits": hits}


def _format_knowledge_context(hits: list[KnowledgeHit]) -> str:
    """把检索片段整理成提示词中的「知识库参考」区块。

    按检索顺序逐条列出 ``chunk_id`` 与 ``text``，便于模型参考正文、也便于
    事后核对实际传入的资料；空列表返回空字符串，调用方据此决定是否拼接。
    """
    if not hits:
        return ""
    blocks = [
        f"[{position}] chunk_id：{hit['chunk_id']}\n{hit['text']}"
        for position, hit in enumerate(hits, start=1)
    ]
    return "知识库参考：\n" + "\n\n".join(blocks)


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
        usage = response.usage_metadata
        output_details = usage.get("output_token_details") if usage else None

        def display(value: object) -> str:
            return "未返回" if value is None else str(value)

        raise RuntimeError(
            "输出被截断，请调整生成长度或创作要求。"
            f"模型：{display(response.response_metadata.get('model_name'))}；"
            f"请求上限：{STORY_MAX_TOKENS} tokens；"
            f"输入 tokens：{display(usage.get('input_tokens') if usage else None)}；"
            f"输出 tokens：{display(usage.get('output_tokens') if usage else None)}；"
            f"推理 tokens：{display(output_details.get('reasoning') if output_details else None)}。"
        )
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

    请求同时携带 ``requirement`` 与 ``story``；状态中有 ``knowledge_hits`` 时，
    在梗概之后附加知识库参考区块，为空或缺失时不加。模型以 ``function_calling``
    的方式按 ``EpisodeScript`` 约束返回结构化结果。原始响应用于检查结束
    原因与读取用量，解析结果与解析错误分别用于取剧本和报错。检查全部通过
    后返回 ``script``；``raw`` 提供了用量时，复制状态中已有的 ``usage_by_node``
    （不存在则按空字典处理），在新字典中登记 ``generate_script`` 的用量后一并返回。
    """
    story = state.get("story")
    if not story:
        raise ValueError("状态缺少 story，请先执行梗概生成节点。")

    knowledge_context = _format_knowledge_context(state.get("knowledge_hits", []))
    content_parts = [
        f"原始创作要求：\n{state['requirement']}",
        f"故事梗概：\n{story}",
    ]
    if knowledge_context:
        content_parts.append(knowledge_context)
    content_parts.append("请依据以上要求与梗概，输出单集剧本的 JSON。")

    messages = [
        SystemMessage(content=SCRIPT_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(content_parts)),
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

    请求同时携带 ``requirement`` 与完整剧本的 JSON 文本；状态中有
    ``knowledge_hits`` 时，在剧本之后附加知识库参考区块，为空或缺失时不加。
    模型以 ``function_calling`` 的方式按 ``EpisodeStoryboardDraft`` 返回结构化
    草稿，草稿不含 ``shot_id``；结束原因、解析错误与解析结果检查通过后，交由
    ``finalize_storyboard`` 分配 UUID 并核对场次覆盖、播放顺序与角色引用。
    成功时返回正式 ``storyboard``；``raw`` 提供了用量时，复制状态中已有的
    ``usage_by_node``（不存在则按空字典处理），登记 ``generate_storyboard``
    的用量后一并返回。
    """
    script = state.get("script")
    if script is None:
        raise ValueError("状态缺少 script，请先执行编剧节点。")

    knowledge_context = _format_knowledge_context(state.get("knowledge_hits", []))
    content_parts = [
        f"原始创作要求：\n{state['requirement']}",
        f"单集剧本：\n{script.model_dump_json(indent=2)}",
    ]
    if knowledge_context:
        content_parts.append(knowledge_context)
    content_parts.append("请依据以上要求与剧本，输出整集分镜的 JSON。")

    messages = [
        SystemMessage(content=STORYBOARD_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(content_parts)),
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


def build_workflow(
    model: BaseChatModel,
    knowledge_index_provider: Callable[[], InMemoryVectorStore],
) -> CompiledStateGraph:
    """构建并编译故事生成工作流。

    构图时把模型客户端与索引提供器固定下来：注册到图中的节点只接收
    LangGraph 传入的状态，再连同对应依赖转交给节点逻辑。执行顺序为
    ``START → retrieve_knowledge_context → generate_story →
    generate_script → generate_storyboard → END``，其中检索节点先依据原始
    需求检索一次知识库，正常完成时三个生成节点各调用一次模型，共三次；
    前一个节点失败时，后一个节点不会执行。索引提供器只保存可调用对象，
    建库延迟到检索节点执行时，因此构图与编译阶段不会加载索引。调用方拿到
    编译后的图，传入包含 ``requirement`` 的初始状态即可执行一次任务。
    """

    def retrieve_knowledge_context_node(
        state: WorkflowState,
    ) -> dict[str, object]:
        """图中的检索节点：把状态与已绑定的索引提供器转交给检索逻辑。"""
        return retrieve_knowledge_context(state, knowledge_index_provider)

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

    builder.add_node("retrieve_knowledge_context", retrieve_knowledge_context_node)
    builder.add_node("generate_story", generate_story_node)
    builder.add_node("generate_script", generate_script_node)
    builder.add_node("generate_storyboard", generate_storyboard_node)

    builder.add_edge(START, "retrieve_knowledge_context")
    builder.add_edge("retrieve_knowledge_context", "generate_story")
    builder.add_edge("generate_story", "generate_script")
    builder.add_edge("generate_script", "generate_storyboard")
    builder.add_edge("generate_storyboard", END)

    graph = builder.compile()
    return graph
