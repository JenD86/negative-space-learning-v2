from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


@dataclass
class EpisodeTrajectory:
    episode_id: str
    generation_id: int
    episode_index: int
    prompt_responses: List[Dict[str, Any]]
    trajectory: Dict[str, Any]
    space_freed_kb: float
    episode_runtime_success: bool
    success: bool
    action_count: int
    container_variation: str
    started_at: str
    completed_at: str
    duration_seconds: float
    partial: bool = False
    error_message: Optional[str] = None
    space_measurements: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    filesystem_groups: List[Dict[str, Any]] = field(default_factory=list)
    measurement_errors: List[str] = field(default_factory=list)
    container_overhead_seconds: Optional[float] = None
    episode_execution_seconds: Optional[float] = None
    total_inference_ms: Optional[float] = None
    inference_call_count: Optional[int] = None
    average_output_tokens_per_second: Optional[float] = None
    inference_duty_cycle: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    cpu_utilization_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EpisodeTrajectory":
        return cls(
            episode_id=str(payload["episode_id"]),
            generation_id=int(payload["generation_id"]),
            episode_index=int(payload["episode_index"]),
            prompt_responses=list(payload.get("prompt_responses", [])),
            trajectory=dict(payload.get("trajectory", {})),
            space_freed_kb=float(payload.get("space_freed_kb", 0.0)),
            episode_runtime_success=bool(payload.get("episode_runtime_success", False)),
            success=bool(payload.get("success", False)),
            action_count=int(payload.get("action_count", 0)),
            container_variation=str(payload.get("container_variation", "")),
            started_at=str(payload.get("started_at", "")),
            completed_at=str(payload.get("completed_at", "")),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            partial=bool(payload.get("partial", False)),
            error_message=payload.get("error_message"),
            space_measurements={
                key: tuple(value)
                for key, value in dict(payload.get("space_measurements", {})).items()
            },
            filesystem_groups=list(payload.get("filesystem_groups", [])),
            measurement_errors=list(payload.get("measurement_errors", [])),
            container_overhead_seconds=_optional_float(
                payload.get("container_overhead_seconds")
            ),
            episode_execution_seconds=_optional_float(
                payload.get("episode_execution_seconds")
            ),
            total_inference_ms=_optional_float(payload.get("total_inference_ms")),
            inference_call_count=_optional_int(payload.get("inference_call_count")),
            average_output_tokens_per_second=_optional_float(
                payload.get("average_output_tokens_per_second")
            ),
            inference_duty_cycle=_optional_float(payload.get("inference_duty_cycle")),
            gpu_utilization_pct=_optional_float(payload.get("gpu_utilization_pct")),
            cpu_utilization_pct=_optional_float(payload.get("cpu_utilization_pct")),
        )


@dataclass
class GenerationData:
    generation_id: int
    all_episodes: List[EpisodeTrajectory] = field(default_factory=list)
    successful_episodes: List[EpisodeTrajectory] = field(default_factory=list)
    failed_episodes: List[EpisodeTrajectory] = field(default_factory=list)
    total_episodes_run: int = 0
    total_successful: int = 0
    total_rows_collected: int = 0
    total_space_freed_kb: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def add_episode(self, episode: EpisodeTrajectory) -> None:
        self.all_episodes.append(episode)
        self.total_episodes_run += 1
        if episode.success:
            self.successful_episodes.append(episode)
            self.total_successful += 1
            self.total_rows_collected += len(episode.prompt_responses)
            self.total_space_freed_kb += episode.space_freed_kb
        else:
            self.failed_episodes.append(episode)

    @property
    def success_rate(self) -> float:
        if self.total_episodes_run == 0:
            return 0.0
        return self.total_successful / self.total_episodes_run

    def get_sft_training_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for episode in self.successful_episodes:
            for prompt_response in episode.prompt_responses:
                rows.append(
                    {
                        **prompt_response,
                        "episode_id": episode.episode_id,
                        "generation_id": self.generation_id,
                        "episode_space_freed_kb": episode.space_freed_kb,
                    }
                )
        return rows

    def to_metadata_dict(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "generation_id": self.generation_id,
            "total_episodes_run": self.total_episodes_run,
            "total_successful": self.total_successful,
            "total_rows_collected": self.total_rows_collected,
            "total_space_freed_kb": self.total_space_freed_kb,
            "success_rate": self.success_rate,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if run_id is not None:
            payload["run_id"] = run_id

        if self.started_at and self.completed_at:
            started_at = datetime.fromisoformat(self.started_at)
            completed_at = datetime.fromisoformat(self.completed_at)
            elapsed_seconds = (completed_at - started_at).total_seconds()
            if elapsed_seconds > 0:
                elapsed_hours = elapsed_seconds / 3600
                payload["episodes_per_hour"] = self.total_episodes_run / elapsed_hours
                payload["successful_rows_per_hour"] = (
                    self.total_rows_collected / elapsed_hours
                )

        episodes_with_output_rate = [
            episode.average_output_tokens_per_second
            for episode in self.all_episodes
            if episode.average_output_tokens_per_second is not None
        ]
        if episodes_with_output_rate:
            payload["average_output_tokens_per_second"] = sum(
                episodes_with_output_rate
            ) / len(episodes_with_output_rate)

        episodes_with_duty_cycle = [
            episode.inference_duty_cycle
            for episode in self.all_episodes
            if episode.inference_duty_cycle is not None
        ]
        if episodes_with_duty_cycle:
            payload["average_inference_duty_cycle"] = sum(
                episodes_with_duty_cycle
            ) / len(episodes_with_duty_cycle)

        episodes_with_gpu_utilization = [
            episode.gpu_utilization_pct
            for episode in self.all_episodes
            if episode.gpu_utilization_pct is not None
        ]
        if episodes_with_gpu_utilization:
            payload["average_gpu_utilization_pct"] = sum(
                episodes_with_gpu_utilization
            ) / len(episodes_with_gpu_utilization)

        episodes_with_cpu_utilization = [
            episode.cpu_utilization_pct
            for episode in self.all_episodes
            if episode.cpu_utilization_pct is not None
        ]
        if episodes_with_cpu_utilization:
            payload["average_cpu_utilization_pct"] = sum(
                episodes_with_cpu_utilization
            ) / len(episodes_with_cpu_utilization)

        container_overheads = [
            episode.container_overhead_seconds
            for episode in self.all_episodes
            if episode.container_overhead_seconds is not None
        ]
        if container_overheads:
            payload["total_container_overhead_seconds"] = sum(container_overheads)

        inference_seconds = [
            episode.total_inference_ms / 1000
            for episode in self.all_episodes
            if episode.total_inference_ms is not None
        ]
        if inference_seconds:
            payload["total_inference_seconds"] = sum(inference_seconds)

        return payload
