import os
from pathlib import Path

from dotenv import load_dotenv

from agent.model_client import create_model_client
from agent.workflow import WorkflowState, build_workflow


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    # 入口负责读取配置，再将配置交给模型接入层
    load_dotenv(PROJECT_DIR / ".env")

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    else:
        api_key=os.getenv("DEEPSEEK_API_KEY")
    
    if not os.getenv("DEEPSEEK_MODEL"):
        raise ValueError("DEEPSEEK_MODEL 环境变量未设置")
    else:
        model_name=os.getenv("DEEPSEEK_MODEL")
    
    if not os.getenv("DEEPSEEK_API_URL"):
        raise ValueError("DEEPSEEK_API_URL 环境变量未设置")
    else:
        api_url=os.getenv("DEEPSEEK_API_URL")
    
    client = create_model_client(api_key, api_url, model_name)

    # 构建工作流，把客户端绑定进图；正常完成时梗概、编剧与分镜节点各调用一次模型
    workflow = build_workflow(client)

    initial_state: WorkflowState = {
        "requirement": (
            "写一个不超过200字的悬疑漫剧故事梗概："
            "深夜只有小王一人的公交车。"
        )
    }

    result = workflow.invoke(initial_state)

    print("生成结果：\n")
    print(result["story"])

    print("\n剧本：")
    print(result["script"].model_dump_json(indent=2))

    print("\n分镜：")
    for scene in result["storyboard"].scenes:
        for index, shot in enumerate(scene.shots, start=1):
            print(f"第 {scene.scene_no} 场第 {index} 镜")
            print(f"  shot_id：{shot.shot_id}")
            print(f"  画面：{shot.visual_description}")
            print(
                "  出镜："
                + ("、".join(shot.character_ids) if shot.character_ids else "空镜头")
            )

    print("\nToken 用量：")
    usage_by_node = result.get("usage_by_node")
    if not usage_by_node:
        print("接口未提供用量信息")
    else:
        for node_name, usage in usage_by_node.items():
            print(f"{node_name}：{usage}")


if __name__ == "__main__":
    main()