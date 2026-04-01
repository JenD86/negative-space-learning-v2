from typing import List, Optional
from pydantic import BaseModel


class AppConfig(BaseModel):
    dev: bool
    model_name: str
    gpu_memory_utilization: float = 0.85
    code_host_cache_path: str
    container_ids: List[str]
    main_container_idx: int

    # These 2 below are mutually exclusive
    dynamic_container: bool
    docker_compose_dir: str

    train_data_save_folder: str

    class PeftConfig(BaseModel):
        base_model_path: str
        checkpoint_path: str
        device: str = "auto"

    peft: Optional[PeftConfig] = None

    class SpecialEGCConfig(BaseModel):
        count: int
        max_retries: int

    special_egc: SpecialEGCConfig

    class StrategyListConfig(BaseModel):
        max_retries: int

    strategy_list: StrategyListConfig

    class StrategyCodeConfig(BaseModel):
        count: int
        max_retries: int

    strategy_code: StrategyCodeConfig

    class EpisodeConfig(BaseModel):
        action_budget: int = 12
        scratchpad_max_chars: int = 10000
        scratchpad_storage_path: Optional[str] = None
        success_threshold_kb: float = 0.0  # ΔR > 0

    episode: Optional[EpisodeConfig] = None
