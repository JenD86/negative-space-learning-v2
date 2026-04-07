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
from src.backend.utils import is_http_ready
from src.backend.vllm import wait_for_gpu_memory_release
from src.helper import generate_readable_run_id, unflatten_toml_dict
from src.observability import MetricsCollector, MetricsGenner
from src.train import convert_adapter_to_gguf, resolve_training_export_format, train_sft
from src.typing.config import AppConfig

SFT_ROWS_FILENAME = "sft_training_rows.jsonl"
SUMMARY_FILENAME = "orchestration_summary.json"
DEFAULT_VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"


def _adapter_dir(adapter_root: Path, generation_id: int) -> Path:
    return adapter_root / f"after_generation_{generation_id}"


def _llama_lora_path(adapter_dir: Path) -> Path:
    return adapter_dir / "adapter.gguf"


def _training_artifact_path(
    artifact_root: Path,
    generation_id: int,
    export_format: str,
) -> Path:
    if export_format == "gguf":
        return artifact_root / f"after_generation_{generation_id}.gguf"
    return _adapter_dir(artifact_root, generation_id)


def _prepare_training_artifact_for_backend(
    config: AppConfig,
    artifact_path: Path,
    export_format: str,
) -> Path:
    backend = config.model_name.strip().split(":", 1)[0]
    if backend != "llama" or export_format != "peft":
        return artifact_path

    if config.training is None:
        raise ValueError("training config is required to prepare llama artifacts")

    return convert_adapter_to_gguf(
        str(artifact_path),
        str(_llama_lora_path(artifact_path)),
        quantize=config.training.gguf_quantize,
    )


def _apply_training_artifact(
    generation_config: AppConfig,
    artifact_path: Path | None,
    export_format: str,
) -> None:
    if artifact_path is None:
        return

    backend = generation_config.model_name.strip().split(":", 1)[0]
    if backend == "vllm":
        if generation_config.vllm is None:
            generation_config.vllm = AppConfig.VllmConfig()

        if export_format == "peft":
            generation_config.vllm.lora_adapter_path = str(artifact_path)
            return

        if export_format == "merged_16bit":
            generation_config.vllm.lora_adapter_path = None
            generation_config.vllm.local_model_path = str(artifact_path)
            return

        raise ValueError("vLLM training loop does not consume GGUF artifacts")

    if backend == "llama":
        if generation_config.llama is None:
            generation_config.llama = AppConfig.LlamaConfig()

        if export_format == "peft":
            generation_config.llama.lora_adapter_path = str(artifact_path)
            return

        if export_format == "gguf":
            generation_config.llama.lora_adapter_path = None
            generation_config.model_name = f"llama:{artifact_path}"
            return

        raise ValueError("llama backend only supports PEFT adapters or GGUF artifacts")

    raise ValueError(
        f"Training loop does not know how to apply '{export_format}' artifacts to backend '{backend}'"
    )


def _served_adapter_path(config: AppConfig) -> str | None:
    if config.vllm is not None:
        return config.vllm.lora_adapter_path
    if config.llama is not None:
        return config.llama.lora_adapter_path
    return None


def _served_training_artifact_path(config: AppConfig) -> str | None:
    backend = config.model_name.strip().split(":", 1)[0]
    if backend == "vllm" and config.vllm is not None:
        return config.vllm.lora_adapter_path or config.vllm.local_model_path
    if (
        backend == "llama"
        and config.llama is not None
        and config.llama.lora_adapter_path
    ):
        return config.llama.lora_adapter_path
    if backend == "llama" and config.model_name.startswith("llama:"):
        return config.model_name.split(":", 1)[1].strip()
    return None


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


def _vllm_server_is_ready(models_url: str = DEFAULT_VLLM_MODELS_URL) -> bool:
    return is_http_ready(models_url)


def _should_wait_for_vllm_gpu_release(
    config: AppConfig,
    generation_id: int,
) -> bool:
    if config.training is None or config.orchestration is None:
        return False

    if generation_id >= config.orchestration.num_generations - 1:
        return False

    if config.training.gpu_wait_timeout_seconds <= 0:
        return False

    backend = config.model_name.strip().split(":", 1)[0]
    if backend != "vllm":
        return False

    return not _vllm_server_is_ready()


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
    served_adapter_dir = _served_adapter_path(config)
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
    generation_root = Path(config.generation.generation_output_dir) / active_run_id
    adapter_root = Path(config.training.adapter_output_dir) / active_run_id
    training_export_format = resolve_training_export_format(
        config.model_name,
        config.training.export_format,
    )

    latest_artifact_path: Path | None = None
    results: list[dict[str, Any]] = []

    for generation_id in range(config.orchestration.num_generations):
        generation_config = config.model_copy(deep=True)
        assert generation_config.generation is not None
        generation_config.generation.generation_output_dir = str(generation_root)
        _apply_training_artifact(
            generation_config,
            latest_artifact_path,
            training_export_format,
        )
        should_wait_for_gpu_release = _should_wait_for_vllm_gpu_release(
            generation_config,
            generation_id,
        )

        generation_dir = run_generation_phase(
            generation_config,
            generation_id,
            active_run_id,
            active_docker_client,
            metrics_collector=metrics_collector,
        )

        training_paths: list[Path] = []
        trained_artifact_path: Path | None = None
        if generation_id < config.orchestration.num_generations - 1:
            if should_wait_for_gpu_release:
                logger.info("Waiting for GPU memory to be released before training")
                wait_for_gpu_memory_release(
                    min_free_memory_fraction=(
                        config.training.gpu_wait_min_free_memory_fraction
                    ),
                    timeout_s=config.training.gpu_wait_timeout_seconds,
                )

            training_paths = _collect_training_window_paths(
                generation_root,
                end_generation_id=generation_id,
                window_size=config.orchestration.training_window_size,
            )
            trained_artifact_path = train_sft(
                base_model=config.training.base_model,
                training_data_paths=[str(path) for path in training_paths],
                output_dir=str(
                    _training_artifact_path(
                        adapter_root,
                        generation_id,
                        training_export_format,
                    )
                ),
                max_seq_length=config.training.max_seq_length,
                max_steps=config.training.max_steps,
                per_device_train_batch_size=config.training.per_device_train_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                learning_rate=config.training.learning_rate,
                warmup_steps=config.training.warmup_steps,
                report_to=config.training.report_to,
                export_format=training_export_format,
                gguf_quantize=config.training.gguf_quantize,
            )
            if trained_artifact_path is None:
                raise RuntimeError("train_sft returned no artifact path")
            latest_artifact_path = _prepare_training_artifact_for_backend(
                config,
                trained_artifact_path,
                training_export_format,
            )

        results.append(
            {
                "generation_id": generation_id,
                "generation_dir": str(generation_dir),
                "served_adapter_dir": _served_adapter_path(generation_config),
                "served_artifact_path": _served_training_artifact_path(
                    generation_config
                ),
                "training_export_format": training_export_format,
                "trained_adapter_dir": (
                    str(trained_artifact_path)
                    if trained_artifact_path is not None
                    and training_export_format == "peft"
                    else None
                ),
                "trained_artifact_path": (
                    str(trained_artifact_path)
                    if trained_artifact_path is not None
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
            "training_export_format": training_export_format,
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
    return _orchestration_summary_path(
        Path(config.generation.generation_output_dir) / run_id
    )


if __name__ == "__main__":
    print(main())
