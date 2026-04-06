from .QwenVllm import QwenVllmGenner
from .config import ServerConfig


class LlamaGenner(QwenVllmGenner):
    def __init__(self, client, config: ServerConfig):
        super().__init__(client, config, identifier="llama")
