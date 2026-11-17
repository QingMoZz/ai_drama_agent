import os
from pathlib import Path

from dotenv import load_dotenv

from agent.cli import build_parser
from agent.model_client import create_model_client
from agent.snapshot_schema import EpisodeSnapshotPayload
from agent.storage import (
    initialize_database,
    load_episode_snapshot,
    save_episode_snapshot,
)
from agent.workflow import WorkflowState, build_workflow


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "episodes.sqlite3"


def print_episode_payload(payload: EpisodeSnapshotPayload) -> None:
    """展示一份整集内容：梗概、剧本、分镜与各节点用量。"""
    print("生成结果：\n")
    print(payload.story)

    print("\n剧本：")
    print(payload.script.model_dump_json(indent=2))

    print("\n分镜：")
    for scene in payload.storyboard.scenes:
        for index, shot in enumerate(scene.shots, start=1):
            print(f"第 {scene.scene_no} 场第 {index} 镜")
            print(f"  shot_id：{shot.shot_id}")
            print(f"  画面：{shot.visual_description}")
            print(
                "  出镜："
                + ("、".join(shot.character_ids) if shot.character_ids else "空镜头")
            )

    print("\nToken 用量：")
    if not payload.usage_by_node:
        print("接口未提供用量信息")
    else:
        for node_name, usage in payload.usage_by_node.items():
            print(f"{node_name}：{usage}")


def run_show(
    db_path: Path,
    episode_id: str,
    revision: int | None = None,
) -> None:
    """展示某一集的最新版本或指定历史版本的快照。"""
    if revision is not None and revision < 1:
        raise ValueError(f"revision 必须大于等于 1，当前为：{revision}")

    if not db_path.exists():
        print(f"数据库不存在：{db_path}")
        print("请先生成内容，或检查 --db-path 是否指向正确的数据库文件。")
        return

    snapshot = load_episode_snapshot(db_path, episode_id, revision)
    if snapshot is None:
        version = "最新版本" if revision is None else f"第 {revision} 版"
        print(f"未找到匹配的快照：集 ID 为 {episode_id}，{version}。")
        return

    print(f"集 ID：{snapshot.episode_id}")
    print(f"版本：{snapshot.revision}")
    print(f"创建时间：{snapshot.created_at}（UTC）")
    print_episode_payload(snapshot.payload)


def run_generate(db_path: Path, episode_id: str, requirement: str) -> None:
    """生成一集内容并保存为新版本。"""
    # 读取配置，再将配置交给模型接入层
    load_dotenv(PROJECT_DIR / ".env")

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    else:
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if not os.getenv("DEEPSEEK_MODEL"):
        raise ValueError("DEEPSEEK_MODEL 环境变量未设置")
    else:
        model_name = os.getenv("DEEPSEEK_MODEL")

    if not os.getenv("DEEPSEEK_API_URL"):
        raise ValueError("DEEPSEEK_API_URL 环境变量未设置")
    else:
        api_url = os.getenv("DEEPSEEK_API_URL")

    client = create_model_client(api_key, api_url, model_name)

    # 执行工作流前准备好存储位置
    db_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(db_path)

    # 构建工作流，把客户端绑定进图；正常完成时梗概、编剧与分镜节点各调用一次模型
    workflow = build_workflow(client)

    initial_state: WorkflowState = {"requirement": requirement}
    result = workflow.invoke(initial_state)

    payload = EpisodeSnapshotPayload(
        requirement=initial_state["requirement"],
        story=result["story"],
        script=result["script"],
        storyboard=result["storyboard"],
        usage_by_node=result.get("usage_by_node", {}),
    )
    revision = save_episode_snapshot(db_path, episode_id, payload)

    print(f"数据库：{db_path}")
    print(f"集 ID：{episode_id}")
    print(f"版本：{revision}")
    print_episode_payload(payload)


def main() -> None:
    """解析命令行参数，分发到生成或查看命令。"""
    parser = build_parser(DEFAULT_DB_PATH)
    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args.db_path, args.episode_id, args.requirement)
    elif args.command == "show":
        run_show(args.db_path, args.episode_id, args.revision)


if __name__ == "__main__":
    main()