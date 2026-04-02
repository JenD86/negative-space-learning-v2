from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class AppConfig(BaseModel):
    dev: bool
    model_name: str
    gpu_memory_utilization: float = 0.85
    code_host_cache_path: str
    container_ids: List[str]
    main_container_idx: int = 0

    dynamic_container: bool = False
    docker_compose_dir: Optional[str] = None

    train_data_save_folder: str

    class PeftConfig(BaseModel):
        base_model_path: str
        checkpoint_path: str
        device: str = "auto"

    peft: Optional[PeftConfig] = None

    class SpecialEGCConfig(BaseModel):
        count: int = 1
        max_retries: int = 3

    special_egc: SpecialEGCConfig = Field(default_factory=SpecialEGCConfig)

    class StrategyListConfig(BaseModel):
        max_retries: int = 3

    strategy_list: StrategyListConfig = Field(default_factory=StrategyListConfig)

    class StrategyCodeConfig(BaseModel):
        count: int = 1
        max_retries: int = 3

    strategy_code: StrategyCodeConfig = Field(default_factory=StrategyCodeConfig)

    class InferenceConfig(BaseModel):
        temperature: float = 0.5
        max_tokens: int = 4096
        timeout: int = 300  # seconds

    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    class ObservabilityConfig(BaseModel):
        enabled: bool = True
        record_inference: bool = True
        record_phases: bool = True
        record_resources: bool = True
        metrics_output_path: Optional[str] = None
        hardware_tags: List[str] = Field(default_factory=list)
        load_tags: List[str] = Field(default_factory=list)

    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    class EpisodeConfig(BaseModel):
        action_budget: int = 12
        scratchpad_max_chars: int = 10000
        scratchpad_storage_path: Optional[str] = None
        success_threshold_kb: float = 0.0  # ΔR > 0

    episode: Optional[EpisodeConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, values):
        if values.get("docker_compose_dir") == "":
            values["docker_compose_dir"] = None
        return values

    @model_validator(mode="after")
    def _check_dynamic_container(self) -> "AppConfig":
        if self.dynamic_container and not self.docker_compose_dir:
            raise ValueError(
                "docker_compose_dir is required when dynamic_container=True"
            )
        return self
