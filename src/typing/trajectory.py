from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
        return payload
