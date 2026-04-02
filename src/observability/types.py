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
    phase: str
    success: bool
    episode_id: Optional[str] = None
    content: Optional[str] = None
    error_message: Optional[str] = None
    usage: Optional[UsageInfo] = None
    latency_ms: float = 0.0
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
