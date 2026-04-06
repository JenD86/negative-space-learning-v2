import argparse
import json
import random
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

import docker
import pydantic
import toml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.container import ContainerManager
from src.helper import generate_readable_run_id, unflatten_toml_dict
from src.observability.types import UtilizationSummary
from src.scratchpad import CrossEpisodeScratchpad
from src.typing.config import AppConfig
from src.typing.training import (
    append_episode_jsonl,
    load_generation_checkpoint,
    save_generation_checkpoint,
    save_generation_training_data,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData

from src.backend import resolve_backend_context
from src.observability import MetricsCollector, MetricsGenner


class _NullProgressBar:
    def __init__(self, total: Optional[int] = None, initial: int = 0) -> None:
        self.total = total
        self.n = initial

    def update(self, increment: int = 1) -> None:
        self.n += increment

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None

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


def _compute_episode_inference_metrics(
    before_summary: Mapping[str, float | int],
    after_summary: Mapping[str, float | int],
) -> tuple[float, int, Optional[float]]:
    total_inference_ms = max(
        0.0,
        after_summary.get("total_latency_ms", 0.0)
        - before_summary.get("total_latency_ms", 0.0),
    )
    inference_call_count = max(
        0,
        int(after_summary.get("inference_calls", 0))
        - int(before_summary.get("inference_calls", 0)),
    )
    total_output_tokens = max(
        0,
        int(after_summary.get("total_output_tokens", 0))
        - int(before_summary.get("total_output_tokens", 0)),
    )
    if total_inference_ms <= 0:
        return total_inference_ms, inference_call_count, None
    return (
        total_inference_ms,
        inference_call_count,
        total_output_tokens / (total_inference_ms / 1000),
    )


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
    started_at = datetime.now().isoformat()
    generation_config = config.generation or AppConfig.GenerationConfig()
    episode_config = config.episode or AppConfig.EpisodeConfig()
    output_dir = Path(generation_config.generation_output_dir)
    episode_id = f"ep_gen{generation_id}_{episode_index:04d}_{int(time.time())}"
    before_summary = metrics_collector.summary() if metrics_collector else {}

    utilization_summary = UtilizationSummary()
    utilization_sampling_started = False

    def _stop_utilization_sampling() -> UtilizationSummary:
        nonlocal utilization_summary, utilization_sampling_started
        if utilization_sampling_started and metrics_collector is not None:
            utilization_summary = metrics_collector.stop_utilization_sampling()
            utilization_sampling_started = False
        return utilization_summary

    if metrics_collector is not None:
        metrics_collector.start_utilization_sampling()
        utilization_sampling_started = True

    container_started_at = time.perf_counter()
    try:
        population_results = container_manager.populate(variation_index)
    except Exception:
        _stop_utilization_sampling()
        raise
    if not population_results:
        _stop_utilization_sampling()
        raise RuntimeError("container_manager.populate returned no results")
    primary_population = population_results[0]
    variation_name = primary_population.variation_name
    expected_kb = primary_population.expected_kb
    try:
        verification = container_manager.verify_population(
            expected_kb,
            generation_config.population_verification_tolerance,
        )
    except Exception:
        _stop_utilization_sampling()
        raise

    if not verification["success"]:
        container_overhead_seconds = time.perf_counter() - container_started_at
        utilization_summary = _stop_utilization_sampling()
        completed_at = datetime.now().isoformat()
        return EpisodeTrajectory(
            episode_id=episode_id,
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
            duration_seconds=container_overhead_seconds,
            error_message="container population verification failed",
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=0.0,
            total_inference_ms=0.0,
            inference_call_count=0,
            inference_duty_cycle=0.0,
            peak_gpu_utilization_pct=utilization_summary.peak_gpu_utilization_pct,
            peak_cpu_utilization_pct=utilization_summary.peak_cpu_utilization_pct,
            avg_gpu_utilization_pct=utilization_summary.avg_gpu_utilization_pct,
            avg_cpu_utilization_pct=utilization_summary.avg_cpu_utilization_pct,
        )

    scratchpad_storage_path = episode_config.scratchpad_storage_path
    if generation_config.reset_scratchpad_between_episodes and scratchpad_storage_path:
        scratchpad_path = Path(scratchpad_storage_path)
        if scratchpad_path.exists():
            scratchpad_path.unlink()
    container_overhead_seconds = time.perf_counter() - container_started_at

    execution_started_at = time.perf_counter()
    try:
        episode_result = run_episode_v2(
            genner,
            docker_client,
            container_manager.get_containers(),
            config,
            run_id=run_id,
            save_incremental=False,
            episode_id=episode_id,
        )
    except Exception as exc:
        episode_execution_seconds = time.perf_counter() - execution_started_at
        utilization_summary = _stop_utilization_sampling()
        after_summary = metrics_collector.summary() if metrics_collector else {}
        total_inference_ms, inference_call_count, average_output_tokens_per_second = (
            _compute_episode_inference_metrics(before_summary, after_summary)
        )
        scratchpad_snapshot_path = _snapshot_episode_scratchpad(
            scratchpad_storage_path,
            output_dir,
            generation_id,
            episode_index,
            episode_id,
        )
        completed_at = datetime.now().isoformat()
        duration_seconds = container_overhead_seconds + episode_execution_seconds
        trajectory: dict[str, Any] = {}
        if scratchpad_snapshot_path is not None:
            trajectory["scratchpad_snapshot_path"] = str(scratchpad_snapshot_path)
        return EpisodeTrajectory(
            episode_id=episode_id,
            generation_id=generation_id,
            episode_index=episode_index,
            prompt_responses=[],
            trajectory=trajectory,
            space_freed_kb=0.0,
            episode_runtime_success=False,
            success=False,
            action_count=0,
            container_variation=variation_name,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            error_message=str(exc),
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=episode_execution_seconds,
            total_inference_ms=total_inference_ms,
            inference_call_count=inference_call_count,
            average_output_tokens_per_second=average_output_tokens_per_second,
            inference_duty_cycle=(
                (total_inference_ms / 1000) / duration_seconds
                if duration_seconds > 0 and total_inference_ms > 0
                else 0.0
            ),
            peak_gpu_utilization_pct=utilization_summary.peak_gpu_utilization_pct,
            peak_cpu_utilization_pct=utilization_summary.peak_cpu_utilization_pct,
            avg_gpu_utilization_pct=utilization_summary.avg_gpu_utilization_pct,
            avg_cpu_utilization_pct=utilization_summary.avg_cpu_utilization_pct,
        )

    episode_execution_seconds = time.perf_counter() - execution_started_at
    utilization_summary = _stop_utilization_sampling()
    after_summary = metrics_collector.summary() if metrics_collector else {}
    total_inference_ms, inference_call_count, average_output_tokens_per_second = (
        _compute_episode_inference_metrics(before_summary, after_summary)
    )
    completed_at = datetime.now().isoformat()
    final_episode_id = episode_result["episode_id"]
    scratchpad_snapshot_path = _snapshot_episode_scratchpad(
        scratchpad_storage_path,
        output_dir,
        generation_id,
        episode_index,
        final_episode_id,
    )
    space_freed_kb = episode_result["space_freed_kb"]
    episode_runtime_success = episode_result["episode_runtime_success"]
    success = space_freed_kb > generation_config.success_threshold_kb
    duration_seconds = container_overhead_seconds + episode_execution_seconds
    trajectory = episode_result["trajectory"].copy()
    if scratchpad_snapshot_path is not None:
        trajectory["scratchpad_snapshot_path"] = str(scratchpad_snapshot_path)
    return EpisodeTrajectory(
        episode_id=final_episode_id,
        generation_id=generation_id,
        episode_index=episode_index,
        prompt_responses=episode_result["prompt_responses"],
        trajectory=trajectory,
        space_freed_kb=space_freed_kb,
        episode_runtime_success=episode_runtime_success,
        success=success,
        action_count=episode_result["action_count"],
        container_variation=variation_name,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        partial=episode_result["partial"],
        error_message=episode_result["error_message"],
        space_measurements=episode_result["space_measurements"],
        filesystem_groups=episode_result["filesystem_groups"],
        measurement_errors=episode_result["measurement_errors"],
        container_overhead_seconds=container_overhead_seconds,
        episode_execution_seconds=episode_execution_seconds,
        total_inference_ms=total_inference_ms,
        inference_call_count=inference_call_count,
        average_output_tokens_per_second=average_output_tokens_per_second,
        inference_duty_cycle=(
            (total_inference_ms / 1000) / duration_seconds
            if duration_seconds > 0 and total_inference_ms > 0
            else 0.0
        ),
        peak_gpu_utilization_pct=utilization_summary.peak_gpu_utilization_pct,
        peak_cpu_utilization_pct=utilization_summary.peak_cpu_utilization_pct,
        avg_gpu_utilization_pct=utilization_summary.avg_gpu_utilization_pct,
        avg_cpu_utilization_pct=utilization_summary.avg_cpu_utilization_pct,
    )


def _make_progress_bar(
    enabled: bool,
    total: Optional[int],
    description: str,
    initial: int = 0,
):
    if not enabled:
        return _NullProgressBar(total=total, initial=initial)
    return tqdm(total=total, desc=description, initial=initial)


def _generation_dir(output_dir: Path, generation_id: int) -> Path:
    return output_dir / f"generation_{generation_id}"


def _scratchpad_snapshot_dir(output_dir: Path, generation_id: int) -> Path:
    return _generation_dir(output_dir, generation_id) / "scratchpad_snapshots"


def _snapshot_episode_scratchpad(
    scratchpad_storage_path: Optional[str],
    output_dir: Path,
    generation_id: int,
    episode_index: int,
    episode_id: str,
) -> Optional[Path]:
    if not scratchpad_storage_path:
        return None

    safe_episode_id = episode_id.replace("/", "_").replace("\\", "_") or "unknown"
    snapshot_path = _scratchpad_snapshot_dir(output_dir, generation_id) / (
        f"episode_{episode_index:04d}_{safe_episode_id}.json"
    )
    return CrossEpisodeScratchpad.snapshot_storage(
        Path(scratchpad_storage_path),
        snapshot_path,
        episode_id=episode_id,
        episode_index=episode_index,
        generation_id=generation_id,
        snapshot_reason="episode_complete",
    )


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
            generation_data = _load_existing_generation_data(
                all_episodes_path, generation_id
            )
            generation_data.started_at = (
                checkpoint.get("started_at") or generation_data.started_at
            )

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
    generation_started_at = time.perf_counter()

    try:
        for episode_index in range(start_episode_index, generation_config.max_episodes):
            if (
                generation_data.total_rows_collected
                >= generation_config.target_successful_rows
            ):
                break

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
            elapsed_hours = (time.perf_counter() - generation_started_at) / 3600
            if elapsed_hours > 0:
                episode_postfix: dict[str, str] = {}
                if episode.average_output_tokens_per_second is not None:
                    episode_postfix["tok/s"] = (
                        f"{episode.average_output_tokens_per_second:.1f}"
                    )
                if episode.inference_duty_cycle is not None:
                    episode_postfix["duty"] = f"{episode.inference_duty_cycle:.0%}"
                if episode.peak_gpu_utilization_pct is not None:
                    episode_postfix["gpu"] = f"{episode.peak_gpu_utilization_pct:.0f}%"
                if episode.peak_cpu_utilization_pct is not None:
                    episode_postfix["cpu"] = f"{episode.peak_cpu_utilization_pct:.0f}%"
                if episode_postfix:
                    episode_progress.set_postfix(episode_postfix)
                rows_progress.set_postfix(
                    {
                        "rows/hr": (
                            f"{generation_data.total_rows_collected / elapsed_hours:.1f}"
                        )
                    }
                )

            if generation_config.checkpoint_every_episode:
                append_episode_jsonl(episode.to_dict(), all_episodes_path)
                save_generation_checkpoint(
                    _build_checkpoint_payload(
                        generation_data, episode_index + 1, run_id
                    ),
                    checkpoint_path,
                )

            completed_episodes = episode_index + 1
            if (
                generation_config.container_rebuild_interval > 0
                and completed_episodes < generation_config.max_episodes
                and completed_episodes % generation_config.container_rebuild_interval
                == 0
            ):
                container_manager.rebuild()
            elif (
                generation_config.container_restart_interval > 0
                and completed_episodes < generation_config.max_episodes
                and completed_episodes % generation_config.container_restart_interval
                == 0
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

    metrics_collector = MetricsCollector.from_config(config, run_id)

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
