from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from src.typing.config import AppConfig

from .claude import setup_claude
from .llama import setup_llama
from .ollama import setup_ollama
from .vllm import setup_vllm


BackendContextFactory = Callable[[AppConfig], AbstractContextManager[Any]]


_BACKEND_CONTEXT_FACTORIES: dict[str, BackendContextFactory] = {
    "claude": setup_claude,
    "llama": setup_llama,
    "vllm": setup_vllm,
}

_DEFERRED_BACKEND_NOTES: dict[str, str] = {
    "qwen-peft": "qwen-peft is stubbed in the backend resolver because get_genner() does not yet implement the qwen-peft backend path.",
}


@contextmanager
def _setup_deferred_backend(app_config: AppConfig):
    model_name = app_config.model_name.strip()
    note = _DEFERRED_BACKEND_NOTES.get(
        model_name,
        f"{model_name} is stubbed in the backend resolver.",
    )
    raise NotImplementedError(note)
    yield None


def get_backend_context_factory(model_name: str) -> BackendContextFactory | None:
    normalized_model_name = model_name.strip()
    if normalized_model_name in _DEFERRED_BACKEND_NOTES:
        return _setup_deferred_backend

    if normalized_model_name in _BACKEND_CONTEXT_FACTORIES:
        return _BACKEND_CONTEXT_FACTORIES[normalized_model_name]

    prefix = normalized_model_name.split(":", 1)[0]
    if prefix in _BACKEND_CONTEXT_FACTORIES:
        return _BACKEND_CONTEXT_FACTORIES[prefix]

    return setup_ollama


def resolve_backend_context(
    app_config: AppConfig,
) -> AbstractContextManager[Any] | None:
    factory = get_backend_context_factory(app_config.model_name)
    if factory is None:
        return None
    return factory(app_config)
