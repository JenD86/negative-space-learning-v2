from .QwenVllm import QwenVllmGenner
from .config import VllmConfig


class LlamaGenner(QwenVllmGenner):
    def __init__(self, client, config: VllmConfig):
        super().__init__(client, config, identifier="llama")
