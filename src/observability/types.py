from dataclasses import dataclass
from typing import Optional


@dataclass
class UsageInfo:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class InferenceResult:
    content: str
    usage: Optional[UsageInfo] = None


@dataclass
class InferenceMetric:
    inference_id: str
    run_id: str
    backend: str
    phase: str
    success: bool
    episode_id: Optional[str] = None
    content: Optional[str] = None
    error_message: Optional[str] = None
    usage: Optional[UsageInfo] = None
    model: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens_per_second: Optional[float] = None
    output_tokens_per_second: Optional[float] = None
    total_tokens_per_second: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    host_memory_mb: Optional[float] = None


@dataclass
class PhaseMetric:
    phase_name: str
    run_id: str
    duration_ms: float
    success: bool
    episode_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class ResourceSnapshot:
    gpu_memory_mb: Optional[float] = None
    host_memory_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    cpu_utilization_pct: Optional[float] = None


@dataclass
class UtilizationSummary:
    peak_gpu_utilization_pct: Optional[float] = None
    peak_cpu_utilization_pct: Optional[float] = None
    avg_gpu_utilization_pct: Optional[float] = None
    avg_cpu_utilization_pct: Optional[float] = None
    sample_count: int = 0
