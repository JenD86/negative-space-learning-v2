from .config import (
    DreamConfig,
    LlamaConfig,
    QwenConfig,
    ServerConfig,
    VllmConfig,
)
from .Base import Genner
from .Qwen import QwenGenner
from .QwenVllm import QwenVllmGenner
from .Llama import LlamaGenner
from .Dream import DreamGenner
from .Claude import ClaudeGenner, ClaudeConfig
from openai import OpenAI
import anthropic

__all__ = ["get_genner"]


class BackendException(Exception):
    pass


class OaiBackendException(Exception):
    pass


class ClaudeBackendException(Exception):
    pass


def get_genner(
    backend: str,
    qwen_config: QwenConfig = QwenConfig(),
    dream_config: DreamConfig = DreamConfig(),
    server_config: ServerConfig | None = None,
    oai_client: OpenAI | None = None,
    claude_config: ClaudeConfig = ClaudeConfig(),
    claude_client: anthropic.Anthropic | None = None,
) -> Genner:
    """
    Get a genner instance based on the backend.

    Args:
        backend: The backend to use.
        qwen_config: Configuration for the Qwen backend.
        dream_config: Configuration for the Dream backend.
        server_config: Configuration for OpenAI-compatible server backends (vllm, llama).
        oai_client: OpenAI client (required for vllm/llama backends).
        claude_config: Configuration for the Claude backend.
        claude_client: Anthropic client (required for Claude backend).

    Returns:
        Genner: The genner instance.
    """
    available_backends = [
        "qwen",
        "qwen-finetuned",
        "qwen-cleanup-merged",
        "vllm",
        "llama",
        "dream",
        "claude",
    ]

    if backend == "qwen":
        return QwenGenner(qwen_config)
    elif backend == "qwen-uncensored":
        qwen_config.model = "qwen-uncensored:latest"
        return QwenGenner(qwen_config)
    elif backend == "qwen-0c68278e2d-25-01-22":
        qwen_config.model = "qwen2.5-coder-1.5b-instruct-0c68278e2d-25-01-22:latest"
        return QwenGenner(qwen_config)
    elif backend == "qwen-finetuned":
        qwen_config.model = "qwen-finetuned"
        return QwenGenner(qwen_config)
    elif backend == "qwen-cleanup-merged":
        qwen_config.model = "qwen-cleanup-merged"
        return QwenGenner(qwen_config)
    elif backend == "vllm":
        if not oai_client:
            raise Exception(
                "Using backend 'vllm', OpenAI client is required for vLLM backend"
            )
        if server_config is None:
            server_config = VllmConfig()
        return QwenVllmGenner(oai_client, server_config)
    elif backend == "llama":
        if not oai_client:
            raise Exception(
                "Using backend 'llama', OpenAI client is required for llama backend"
            )
        if server_config is None:
            server_config = LlamaConfig()
        return LlamaGenner(oai_client, server_config)
    elif backend == "dream":
        return DreamGenner(dream_config)
    elif backend == "claude":
        if not claude_client:
            raise ClaudeBackendException(
                "Using backend 'claude', Anthropic client is required for Claude backend"
            )
        return ClaudeGenner(claude_client, claude_config)

    raise BackendException(
        f"Unsupported backend: {backend}, available backends: {', '.join(available_backends)}"
    )
