from contextlib import contextmanager
from typing import Iterator

from src.genner import get_genner
from src.genner.config import QwenConfig
from src.typing.config import AppConfig

from .session import BackendSession


@contextmanager
def setup_ollama(app_config: AppConfig) -> Iterator[BackendSession]:
    qwen_config = QwenConfig()
    qwen_config.model = app_config.model_name
    genner = get_genner("qwen", qwen_config=qwen_config)
    yield BackendSession(
        genner=genner,
        client=genner.client,
        config=qwen_config,
    )


__all__ = ["setup_ollama"]
