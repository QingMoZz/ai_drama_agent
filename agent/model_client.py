"""模型接入层：负责创建和配置 DeepSeek 模型客户端。"""

from langchain_deepseek import ChatDeepSeek

def create_model_client(
    api_key: str,
    api_url: str,
    model: str,
) -> ChatDeepSeek:
    """创建 DeepSeek 模型客户端。"""
    api_key = api_key.strip()
    api_url = api_url.strip()
    model_name = model.strip()

    if not api_key:
        raise ValueError("API Key 不能为空")
    
    if not api_url:
        raise ValueError("API URL 不能为空")
    
    if not model_name:
        raise ValueError("模型名称不能为空")

    return ChatDeepSeek(
        api_key=api_key,
        base_url=api_url,
        model_name=model_name,
        max_tokens=1024,
        timeout=60,
        max_retries=0,
        extra_body={
            "thinking": {"type": "disabled"},
        },
    )