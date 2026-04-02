from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from src.typing.config import AppConfig

from .llama import setup_llama


BackendContextFactory = Callable[[AppConfig], AbstractContextManager[Any]]


_BACKEND_CONTEXT_FACTORIES: dict[str, BackendContextFactory] = {
    "llama": setup_llama,
}


def get_backend_context_factory(model_name: str) -> BackendContextFactory | None:
    prefix = model_name.strip().split(":", 1)[0]
    return _BACKEND_CONTEXT_FACTORIES.get(prefix)


def resolve_backend_context(
    app_config: AppConfig,
) -> AbstractContextManager[Any] | None:
    factory = get_backend_context_factory(app_config.model_name)
    if factory is None:
        return None
    return factory(app_config)
