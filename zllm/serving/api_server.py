"""OpenAI 兼容 API 服务器。

基于 FastAPI，提供 /v1/chat/completions 和 /v1/models 端点。
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse


@dataclass
class ChatCompletionRequest:
    model: str = "zllm"
    messages: List[dict] = field(default_factory=list)
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False


def create_app():
    """创建 FastAPI 应用。"""
    app = FastAPI(title="ZLLM API", version="0.1.0")

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "zllm",
                    "object": "model",
                    "owned_by": "zllm",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.85)
        top_p = request.get("top_p", 0.95)
        max_tokens = request.get("max_tokens", 512)
        model_name = request.get("model", "zllm")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app
