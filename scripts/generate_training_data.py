import argparse
import json
import random
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import docker
import pydantic
import toml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.container import ContainerManager
from src.helper import generate_readable_run_id, unflatten_toml_dict
from src.typing.config import AppConfig
from src.typing.training import (
    append_episode_jsonl,
    load_generation_checkpoint,
    save_generation_checkpoint,
    save_generation_training_data,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData


class _NullProgressBar:
    def __init__(self, total: Optional[int] = None, initial: int = 0) -> None:
        self.total = total
        self.n = initial

    def update(self, increment: int = 1) -> None:
        self.n += increment

    def close(self) -> None:
        return None


def run_episode_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from scripts.main import run_episode_v2 as inner_run_episode_v2

    return inner_run_episode_v2(*args, **kwargs)


def select_variation_index(
    strategy: str,
    episode_index: int,
    variation_count: int,
    rng: Optional[random.Random] = None,
) -> int:
    if variation_count <= 0:
        raise ValueError("variation_count must be positive")
    if strategy == "round_robin":
        return episode_index % variation_count
    if strategy == "random":
        active_rng = rng or random.Random()
        return active_rng.randrange(variation_count)
    raise ValueError(f"Unsupported variation strategy: {strategy}")


def run_single_episode(
    genner,
    docker_client,
    container_manager,
    config,
    generation_id,
    episode_index,
    variation_index,
    run_id,
    metrics_collector=None,
) -> EpisodeTrajectory:
    del metrics_collector
    started_at = datetime.now().isoformat()
    generation_config = config.generation or AppConfig.GenerationConfig()
    episode_config = config.episode or AppConfig.EpisodeConfig()
    population_results = container_manager.populate(variation_index)
    primary_population = population_results[0] if population_results else None
    variation_name = getattr(primary_population, "variation_name", f"variation_{variation_index}")
    expected_kb = getattr(primary_population, "expected_kb", 0)
    verification = container_manager.verify_population(
        expected_kb,
        generation_config.population_verification_tolerance,
    )

    if not verification.get("success", False):
        completed_at = datetime.now().isoformat()
        return EpisodeTrajectory(
            episode_id=f"ep_gen{generation_id}_{episode_index}",
            generation_id=generation_id,
            episode_index=episode_index,
            prompt_responses=[],
            trajectory={},
            space_freed_kb=0.0,
            episode_runtime_success=False,
            success=False,
            action_count=0,
            container_variation=variation_name,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(
                datetime.fromisoformat(completed_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds(),
            error_message="container population verification failed",
        )

    scratchpad_storage_path = episode_config.scratchpad_storage_path
    if (
        generation_config.reset_scratchpad_between_episodes
        and scratchpad_storage_path is not None
    ):
        scratchpad_path = Path(scratchpad_storage_path)
        if scratchpad_path.exists():
            scratchpad_path.unlink()

    try:
        episode_result = run_episode_v2(
            genner,
            docker_client,
            container_manager.get_containers(),
            config,
            run_id=run_id,
            save_incremental=False,
        )
    except Exception as exc:
        completed_at = datetime.now().isoformat()
        return EpisodeTrajectory(
            episode_id=f"ep_gen{generation_id}_{episode_index}",
            generation_id=generation_id,
            episode_index=episode_index,
            prompt_responses=[],
            trajectory={},
            space_freed_kb=0.0,
            episode_runtime_success=False,
            success=False,
            action_count=0,
            container_variation=variation_name,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(
                datetime.fromisoformat(completed_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds(),
            error_message=str(exc),
        )

    completed_at = datetime.now().isoformat()
    space_freed_kb = float(episode_result.get("space_freed_kb", 0.0))
    episode_runtime_success = bool(
        episode_result.get("episode_runtime_success", episode_result.get("success", False))
    )
    success = space_freed_kb > generation_config.success_threshold_kb
    return EpisodeTrajectory(
        episode_id=str(episode_result.get("episode_id", f"ep_gen{generation_id}_{episode_index}")),
        generation_id=generation_id,
        episode_index=episode_index,
        prompt_responses=list(episode_result.get("prompt_responses", [])),
        trajectory=dict(episode_result.get("trajectory", {})),
        space_freed_kb=space_freed_kb,
        episode_runtime_success=episode_runtime_success,
        success=success,
        action_count=int(episode_result.get("action_count", 0)),
        container_variation=variation_name,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=(
            datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
        ).total_seconds(),
        partial=bool(episode_result.get("partial", False)),
        error_message=episode_result.get("error_message"),
        space_measurements=dict(episode_result.get("space_measurements", {})),
        filesystem_groups=list(episode_result.get("filesystem_groups", [])),
        measurement_errors=list(episode_result.get("measurement_errors", [])),
    )


def _make_progress_bar(
    enabled: bool,
    total: Optional[int],
    description: str,
    initial: int = 0,
):
    if not enabled:
        return _NullProgressBar(total=total, initial=initial)
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgressBar(total=total, initial=initial)
    return tqdm(total=total, desc=description, initial=initial)


def _generation_dir(output_dir: Path, generation_id: int) -> Path:
    return output_dir / f"generation_{generation_id}"


def _load_existing_generation_data(
    all_episodes_path: Path,
    generation_id: int,
) -> GenerationData:
    generation_data = GenerationData(generation_id=generation_id)
    if not all_episodes_path.exists():
        return generation_data
    with open(all_episodes_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            generation_data.add_episode(EpisodeTrajectory.from_dict(json.loads(line)))
    return generation_data


def _build_checkpoint_payload(
    generation_data: GenerationData,
    next_episode_index: int,
    run_id: str,
) -> dict[str, Any]:
    payload = generation_data.to_metadata_dict(run_id=run_id)
    payload["next_episode_index"] = next_episode_index
    return payload


def run_generation(
    genner,
    docker_client,
    config,
    generation_id,
    run_id,
    metrics_collector=None,
) -> GenerationData:
    generation_config = config.generation or AppConfig.GenerationConfig()
    output_dir = Path(generation_config.generation_output_dir)
    generation_dir = _generation_dir(output_dir, generation_id)
    checkpoint_path = generation_dir / "checkpoint.json"
    all_episodes_path = generation_dir / "all_episodes.jsonl"

    start_episode_index = 0
    generation_data = GenerationData(generation_id=generation_id)
    if generation_config.resume_from_checkpoint:
        checkpoint = load_generation_checkpoint(checkpoint_path)
        if checkpoint is not None:
            start_episode_index = int(checkpoint.get("next_episode_index", 0))
            generation_data = _load_existing_generation_data(all_episodes_path, generation_id)
            generation_data.started_at = checkpoint.get("started_at") or generation_data.started_at

    if generation_data.started_at is None:
        generation_data.started_at = datetime.now().isoformat()

    container_manager = ContainerManager(
        docker_client=docker_client,
        container_ids=config.container_ids,
        docker_compose_dir=config.docker_compose_dir,
        post_rebuild_wait_seconds=generation_config.post_rebuild_wait_seconds,
    )
    try:
        container_manager.verify_ready()
    except Exception:
        if config.dynamic_container and config.docker_compose_dir:
            container_manager.rebuild()
        else:
            raise

    variation_count = len(container_manager.get_mixed_cleanup_variations())
    rng = (
        random.Random(generation_config.variation_random_seed)
        if generation_config.variation_strategy == "random"
        else None
    )
    episode_progress = _make_progress_bar(
        generation_config.show_progress,
        generation_config.max_episodes,
        "episodes",
        initial=generation_data.total_episodes_run,
    )
    rows_progress = _make_progress_bar(
        generation_config.show_progress,
        generation_config.target_successful_rows,
        "successful rows",
        initial=generation_data.total_rows_collected,
    )

    try:
        for episode_index in range(start_episode_index, generation_config.max_episodes):
            if generation_data.total_rows_collected >= generation_config.target_successful_rows:
                break
            if (
                metrics_collector is not None
                and generation_config.resource_snapshot_interval_episodes > 0
                and episode_index % generation_config.resource_snapshot_interval_episodes == 0
            ):
                metrics_collector.snapshot_resources()

            variation_index = select_variation_index(
                generation_config.variation_strategy,
                episode_index,
                variation_count,
                rng=rng,
            )
            previous_rows = generation_data.total_rows_collected
            episode = run_single_episode(
                genner=genner,
                docker_client=docker_client,
                container_manager=container_manager,
                config=config,
                generation_id=generation_id,
                episode_index=episode_index,
                variation_index=variation_index,
                run_id=run_id,
                metrics_collector=metrics_collector,
            )
            generation_data.add_episode(episode)
            episode_progress.update(1)
            rows_progress.update(generation_data.total_rows_collected - previous_rows)

            if generation_config.checkpoint_every_episode:
                append_episode_jsonl(episode.to_dict(), all_episodes_path)
                save_generation_checkpoint(
                    _build_checkpoint_payload(generation_data, episode_index + 1, run_id),
                    checkpoint_path,
                )

            completed_episodes = episode_index + 1
            if (
                generation_config.container_rebuild_interval > 0
                and completed_episodes < generation_config.max_episodes
                and completed_episodes % generation_config.container_rebuild_interval == 0
            ):
                container_manager.rebuild()
            elif (
                generation_config.container_restart_interval > 0
                and completed_episodes < generation_config.max_episodes
                and completed_episodes % generation_config.container_restart_interval == 0
            ):
                container_manager.restart()
    finally:
        episode_progress.close()
        rows_progress.close()

    generation_data.completed_at = datetime.now().isoformat()
    return generation_data


def save_generation_data(
    generation_data,
    output_dir,
    run_id,
) -> Path:
    output_dir = Path(output_dir)
    generation_dir = _generation_dir(output_dir, generation_data.generation_id)
    generation_dir.mkdir(parents=True, exist_ok=True)
    successful_dir = generation_dir / "successful"
    failed_dir = generation_dir / "failed"
    successful_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = generation_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            generation_data.to_metadata_dict(run_id=run_id),
            handle,
            indent=2,
            default=str,
        )

    save_generation_training_data(
        generation_data.get_sft_training_rows(),
        generation_dir / "sft_training_rows.jsonl",
    )

    all_episodes_path = generation_dir / "all_episodes.jsonl"
    with open(all_episodes_path, "w", encoding="utf-8") as handle:
        for episode in generation_data.all_episodes:
            handle.write(json.dumps(episode.to_dict(), default=str) + "\n")

    save_generation_checkpoint(
        _build_checkpoint_payload(
            generation_data,
            generation_data.total_episodes_run,
            run_id,
        ),
        generation_dir / "checkpoint.json",
    )

    for episode in generation_data.successful_episodes:
        with open(
            successful_dir / f"{episode.episode_id}.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(episode.to_dict(), handle, indent=2, default=str)

    for episode in generation_data.failed_episodes:
        with open(
            failed_dir / f"{episode.episode_id}.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(episode.to_dict(), handle, indent=2, default=str)

    return generation_dir


def _build_metrics_collector(config: AppConfig, run_id: str):
    from src.observability import MetricsCollector

    output_dir = config.observability.metrics_output_path or config.train_data_save_folder
    return MetricsCollector(
        run_id=run_id,
        output_dir=output_dir,
        enabled=config.observability.enabled,
        record_inference=config.observability.record_inference,
        record_phases=config.observability.record_phases,
        record_resources=config.observability.record_resources,
    )


def main() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config-generation.toml")
    parser.add_argument("--generation-id", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config_dict = toml.load(handle)
    try:
        config = AppConfig(**unflatten_toml_dict(config_dict))
    except pydantic.ValidationError as exc:
        raise RuntimeError(f"Config validation error: {exc}") from exc

    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()
    if args.target_rows is not None:
        config.generation.target_successful_rows = args.target_rows

    run_id = generate_readable_run_id()
    metrics_collector = _build_metrics_collector(config, run_id)

    from src.backend import resolve_backend_context
    from src.observability import MetricsGenner

    docker_client = docker.from_env()
    backend_context = resolve_backend_context(config)
    if backend_context is None:
        raise RuntimeError(f"No backend context registered for {config.model_name}")

    with ExitStack() as backend_stack:
        backend_session = backend_stack.enter_context(backend_context)
        smoke_test = getattr(backend_session, "smoke_test", None)
        if smoke_test is not None:
            logger.info(f"Inference test response: {smoke_test()}")
        genner = MetricsGenner(backend_session.genner, metrics_collector)
        generation_data = run_generation(
            genner=genner,
            docker_client=docker_client,
            config=config,
            generation_id=args.generation_id,
            run_id=run_id,
            metrics_collector=metrics_collector,
        )

    output_path = save_generation_data(
        generation_data,
        Path(config.generation.generation_output_dir),
        run_id,
    )
    metrics_collector.flush()
    return output_path


if __name__ == "__main__":
    main()
