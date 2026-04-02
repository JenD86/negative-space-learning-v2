import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from loguru import logger

from src.observability.types import InferenceMetric, PhaseMetric, ResourceSnapshot


class MetricsCollector:
    def __init__(
        self,
        run_id: str,
        output_dir: str | Path,
        enabled: bool = True,
        record_inference: bool = True,
        record_phases: bool = True,
        record_resources: bool = True,
    ) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.record_inference_enabled = record_inference
        self.record_phases_enabled = record_phases
        self.record_resources_enabled = record_resources
        self.inference_metrics: list[InferenceMetric] = []
        self.phase_metrics: list[PhaseMetric] = []
        self._lock = threading.Lock()
        self._summary = {
            "inference_calls": 0,
            "phase_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency_ms": 0.0,
            "generation_count": 0,
            "_prompt_tokens_per_second_sum": 0.0,
            "_output_tokens_per_second_sum": 0.0,
            "_total_tokens_per_second_sum": 0.0,
        }

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"metrics_{self.run_id}.jsonl"

    def record_inference(self, metric: InferenceMetric) -> None:
        if not self.enabled or not self.record_inference_enabled:
            return

        with self._lock:
            self.inference_metrics.append(metric)
            self._summary["inference_calls"] += 1
            self._summary["total_latency_ms"] += metric.latency_ms
            if metric.usage is not None:
                self._summary["total_input_tokens"] += metric.usage.prompt_tokens or 0
                self._summary["total_output_tokens"] += (
                    metric.usage.completion_tokens or 0
                )
            if metric.total_tokens_per_second is not None:
                self._summary["generation_count"] += 1
                self._summary["_prompt_tokens_per_second_sum"] += (
                    metric.prompt_tokens_per_second or 0.0
                )
                self._summary["_output_tokens_per_second_sum"] += (
                    metric.output_tokens_per_second or 0.0
                )
                self._summary["_total_tokens_per_second_sum"] += (
                    metric.total_tokens_per_second or 0.0
                )

    def record_inference_safe(self, metric: InferenceMetric) -> None:
        try:
            self.record_inference(metric)
        except Exception as exc:
            logger.error(f"Failed to record inference metric: {exc}")

    def record_phase(self, metric: PhaseMetric) -> None:
        if not self.enabled or not self.record_phases_enabled:
            return

        with self._lock:
            self.phase_metrics.append(metric)
            self._summary["phase_count"] += 1

    def record_phase_safe(self, metric: PhaseMetric) -> None:
        try:
            self.record_phase(metric)
        except Exception as exc:
            logger.error(f"Failed to record phase metric: {exc}")

    def snapshot_resources(self) -> ResourceSnapshot:
        if not self.enabled or not self.record_resources_enabled:
            return ResourceSnapshot()

        return ResourceSnapshot(
            gpu_memory_mb=self._read_gpu_memory_mb(),
            host_memory_mb=self._read_host_memory_mb(),
        )

    def flush(self) -> Optional[str]:
        if not self.enabled:
            with self._lock:
                self.inference_metrics.clear()
                self.phase_metrics.clear()
            return None

        with self._lock:
            inference_metrics = list(self.inference_metrics)
            phase_metrics = list(self.phase_metrics)
            self.inference_metrics.clear()
            self.phase_metrics.clear()

        if not inference_metrics and not phase_metrics:
            return str(self.output_path) if self.output_path.exists() else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("a", encoding="utf-8") as handle:
            for payload in self._serialize_metrics(inference_metrics, phase_metrics):
                line = json.dumps(payload, default=str)
                if not line:
                    raise ValueError("Refusing to write an empty metrics line")
                json.loads(line)
                handle.write(line + "\n")

        return str(self.output_path)

    def summary(self) -> dict[str, float | int]:
        with self._lock:
            summary = dict(self._summary)

        generation_count = int(summary["generation_count"])
        if generation_count > 0:
            summary["average_prompt_tokens_per_second"] = (
                float(summary["_prompt_tokens_per_second_sum"]) / generation_count
            )
            summary["average_output_tokens_per_second"] = (
                float(summary["_output_tokens_per_second_sum"]) / generation_count
            )
            summary["average_total_tokens_per_second"] = (
                float(summary["_total_tokens_per_second_sum"]) / generation_count
            )
        else:
            summary["average_prompt_tokens_per_second"] = 0.0
            summary["average_output_tokens_per_second"] = 0.0
            summary["average_total_tokens_per_second"] = 0.0

        del summary["_prompt_tokens_per_second_sum"]
        del summary["_output_tokens_per_second_sum"]
        del summary["_total_tokens_per_second_sum"]

        return summary

    def _serialize_metrics(
        self,
        inference_metrics: list[InferenceMetric],
        phase_metrics: list[PhaseMetric],
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for metric in inference_metrics:
            payload = asdict(metric)
            payload["metric_type"] = "inference"
            payloads.append(payload)
        for metric in phase_metrics:
            payload = asdict(metric)
            payload["metric_type"] = "phase"
            payloads.append(payload)
        return payloads

    def _read_gpu_memory_mb(self) -> Optional[float]:
        try:
            import torch
        except ImportError:
            return None

        if not torch.cuda.is_available():
            return None

        return float(torch.cuda.memory_allocated() / (1024 * 1024))

    def _read_host_memory_mb(self) -> Optional[float]:
        try:
            import psutil
        except ImportError:
            psutil = None

        if psutil is not None:
            return float(psutil.Process().memory_info().rss / (1024 * 1024))

        status_path = Path("/proc/self/status")
        if not status_path.exists():
            return None

        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024

        return None
