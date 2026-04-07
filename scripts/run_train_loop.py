from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import docker
import pydantic
import toml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_training_data import run_generation, save_generation_data
from src.backend import resolve_backend_context
from src.helper import generate_readable_run_id, unflatten_toml_dict
from src.observability import MetricsCollector, MetricsGenner
from src.train import train_sft
from src.typing.config import AppConfig

SFT_ROWS_FILENAME = "sft_training_rows.jsonl"
SUMMARY_FILENAME = "orchestration_summary.json"


def _adapter_dir(adapter_root: Path, generation_id: int) -> Path:
    return adapter_root / f"after_generation_{generation_id}"


def _collect_training_window_paths(
    generation_root: Path,
    end_generation_id: int,
    window_size: int,
) -> list[Path]:
    if window_size < 1:
        raise ValueError("training window size must be positive")

    start_generation_id = max(0, end_generation_id - window_size + 1)
    training_paths: list[Path] = []
    for generation_id in range(start_generation_id, end_generation_id + 1):
        training_path = (
            generation_root / f"generation_{generation_id}" / SFT_ROWS_FILENAME
        )
        if not training_path.exists():
            raise FileNotFoundError(f"Training data not found: {training_path}")
        training_paths.append(training_path)
    return training_paths


def _orchestration_summary_path(generation_root: Path) -> Path:
    return generation_root / SUMMARY_FILENAME


def _write_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_generation_phase(
    config: AppConfig,
    generation_id: int,
    run_id: str,
    docker_client: Any,
    metrics_collector: MetricsCollector | None = None,
) -> Path:
    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()

    backend_context = resolve_backend_context(config)
    if backend_context is None:
        raise RuntimeError(f"No backend context registered for {config.model_name}")

    served_model_name = (
        config.vllm.served_model_name
        if config.vllm and config.vllm.served_model_name
        else config.model_name
    )
    served_adapter_dir = config.vllm.lora_adapter_path if config.vllm else None
    logger.info(
        f"Starting generation {generation_id} with model={served_model_name} "
        f"adapter={served_adapter_dir or '<base>'}"
    )

    with ExitStack() as backend_stack:
        backend_session = backend_stack.enter_context(backend_context)
        smoke_test = getattr(backend_session, "smoke_test", None)
        if callable(smoke_test):
            logger.info(f"Inference test response: {smoke_test()}")

        genner = backend_session.genner
        if metrics_collector is not None:
            genner = MetricsGenner(genner, metrics_collector)

        generation_data = run_generation(
            genner=genner,
            docker_client=docker_client,
            config=config,
            generation_id=generation_id,
            run_id=run_id,
            metrics_collector=metrics_collector,
        )

    generation_dir = save_generation_data(
        generation_data,
        Path(config.generation.generation_output_dir),
        run_id,
    )
    if metrics_collector is not None:
        metrics_collector.flush()
    return generation_dir


def run_loop(
    config: AppConfig,
    *,
    run_id: str | None = None,
    docker_client: Any = None,
    metrics_collector: MetricsCollector | None = None,
) -> list[dict[str, Any]]:
    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()
    if config.training is None:
        config.training = AppConfig.TrainingConfig()
    if config.orchestration is None:
        config.orchestration = AppConfig.OrchestrationConfig()

    if config.orchestration.num_generations < 1:
        raise ValueError("num_generations must be at least 1")
    if config.orchestration.training_window_size < 1:
        raise ValueError("training_window_size must be at least 1")

    active_run_id = run_id or generate_readable_run_id()
    active_docker_client = (
        docker_client if docker_client is not None else docker.from_env()
    )
    generation_root = Path(config.generation.generation_output_dir)
    adapter_root = Path(config.training.adapter_output_dir)

    latest_adapter_dir: Path | None = None
    results: list[dict[str, Any]] = []

    for generation_id in range(config.orchestration.num_generations):
        generation_config = config.model_copy(deep=True)
        if generation_config.vllm is None:
            generation_config.vllm = AppConfig.VllmConfig()
        generation_config.vllm.lora_adapter_path = (
            str(latest_adapter_dir) if latest_adapter_dir is not None else None
        )

        generation_dir = run_generation_phase(
            generation_config,
            generation_id,
            active_run_id,
            active_docker_client,
            metrics_collector=metrics_collector,
        )

        training_paths: list[Path] = []
        trained_adapter_dir: Path | None = None
        if generation_id < config.orchestration.num_generations - 1:
            training_paths = _collect_training_window_paths(
                generation_root,
                end_generation_id=generation_id,
                window_size=config.orchestration.training_window_size,
            )
            trained_adapter_dir = train_sft(
                base_model=config.training.base_model,
                training_data_paths=[str(path) for path in training_paths],
                output_dir=str(_adapter_dir(adapter_root, generation_id)),
                max_seq_length=config.training.max_seq_length,
                max_steps=config.training.max_steps,
                per_device_train_batch_size=config.training.per_device_train_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                learning_rate=config.training.learning_rate,
                warmup_steps=config.training.warmup_steps,
                report_to=config.training.report_to,
            )
            latest_adapter_dir = trained_adapter_dir

        results.append(
            {
                "generation_id": generation_id,
                "generation_dir": str(generation_dir),
                "served_adapter_dir": generation_config.vllm.lora_adapter_path,
                "trained_adapter_dir": (
                    str(trained_adapter_dir)
                    if trained_adapter_dir is not None
                    else None
                ),
                "training_data_paths": [str(path) for path in training_paths],
            }
        )

    _write_summary(
        _orchestration_summary_path(generation_root),
        {
            "run_id": active_run_id,
            "num_generations": config.orchestration.num_generations,
            "training_window_size": config.orchestration.training_window_size,
            "generations": results,
        },
    )
    return results


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config-test-loop.toml")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as handle:
        config_dict = toml.load(handle)
    try:
        config = AppConfig(**unflatten_toml_dict(config_dict))
    except pydantic.ValidationError as exc:
        raise RuntimeError(f"Config validation error: {exc}") from exc

    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()

    run_id = generate_readable_run_id()
    metrics_collector = MetricsCollector.from_config(config, run_id)
    run_loop(config, run_id=run_id, metrics_collector=metrics_collector)
    return _orchestration_summary_path(Path(config.generation.generation_output_dir))


if __name__ == "__main__":
    print(main())
