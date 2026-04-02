import time
from typing import Callable, Optional

from loguru import logger
from result import Err, Ok, Result

from src.genner.Base import Genner
from src.helper import nanoid
from src.observability.collector import MetricsCollector
from src.observability.types import InferenceMetric, InferenceResult, UsageInfo
from src.typing.message import Message


class MetricsGenner(Genner):
    def __init__(
        self,
        inner: Genner,
        collector: MetricsCollector,
        phase_name_provider: Optional[Callable[[], Optional[str]]] = None,
        episode_id_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        super().__init__(inner.identifier)
        self.inner = inner
        self.collector = collector
        self.phase_name_provider = phase_name_provider
        self.episode_id_provider = episode_id_provider

    def plist_completion(self, messages: list[Message]) -> Result[InferenceResult, str]:
        inference_id = nanoid()
        phase_name = self._resolve_phase_name(messages)
        episode_id = self._resolve_episode_id(messages)
        started_at = time.perf_counter()

        with logger.contextualize(inference_id=inference_id):
            result = self.inner.plist_completion(messages)

        latency_ms = (time.perf_counter() - started_at) * 1000
        resources = self.collector.snapshot_resources()

        match result:
            case Ok(inference_result):
                self.collector.record_inference_safe(
                    InferenceMetric(
                        inference_id=inference_id,
                        run_id=self.collector.run_id,
                        episode_id=episode_id,
                        phase=phase_name,
                        success=True,
                        content=inference_result.content,
                        usage=inference_result.usage,
                        latency_ms=latency_ms,
                        gpu_memory_mb=resources.gpu_memory_mb,
                        host_memory_mb=resources.host_memory_mb,
                    )
                )
                return Ok(inference_result)
            case Err(error_message):
                self.collector.record_inference_safe(
                    InferenceMetric(
                        inference_id=inference_id,
                        run_id=self.collector.run_id,
                        episode_id=episode_id,
                        phase=phase_name,
                        success=False,
                        error_message=error_message,
                        latency_ms=latency_ms,
                        gpu_memory_mb=resources.gpu_memory_mb,
                        host_memory_mb=resources.host_memory_mb,
                    )
                )
                return Err(error_message)

    def generate_code(self, messages: list[Message]):
        return self.inner.generate_code(messages)

    def generate_list(self, messages: list[Message]):
        return self.inner.generate_list(messages)

    def extract_code(self, response: str):
        return self.inner.extract_code(response)

    def extract_list(self, response: str):
        return self.inner.extract_list(response)

    @staticmethod
    def get_usage_info(response) -> UsageInfo:
        return UsageInfo()

    def _resolve_phase_name(self, messages: list[Message]) -> str:
        for message in messages:
            meta = message.get("meta")
            if isinstance(meta, dict):
                phase_name = meta.get("phase")
                if isinstance(phase_name, str) and phase_name:
                    return phase_name

        if self.phase_name_provider is not None:
            phase_name = self.phase_name_provider()
            if isinstance(phase_name, str) and phase_name:
                return phase_name

        return "unknown"

    def _resolve_episode_id(self, messages: list[Message]) -> Optional[str]:
        for message in messages:
            meta = message.get("meta")
            if isinstance(meta, dict):
                episode_id = meta.get("episode_id")
                if isinstance(episode_id, str) and episode_id:
                    return episode_id

        if self.episode_id_provider is None:
            return None

        episode_id = self.episode_id_provider()
        if isinstance(episode_id, str) and episode_id:
            return episode_id

        return None
