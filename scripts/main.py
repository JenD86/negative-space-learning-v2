import atexit
import shutil
import subprocess
import sys
import json
import random
import platform
import tempfile
from contextlib import ExitStack
from datetime import datetime
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from typing import Any, Dict, List, Optional, Tuple

import pydantic
from loguru import logger
from result import Err, Ok
from docker.models.containers import Container as DockerContainer

import docker
import toml
from src.agent import (
    generate_special_environment_getter_code,
    generate_strategy_code,
    generate_strategy_list,
    regenerate_code,
    regenerate_list,
    regenerate_env_discovery_code,
)
from src.backend import resolve_backend_context
from src.genner import get_genner
from src.genner.Base import Genner
from src.helper import (
    generate_readable_run_id,
    get_formatted_repo_info,
    string_hash,
    unflatten_toml_dict,
)
from src.observability import MetricsCollector, MetricsGenner, PhaseMetric
from src.tool.code import validate_code_offline
from src.tool.docker import (
    calculate_deduplicated_space_freed,
    get_container_free_disk_space_kb_v1,
    get_container_free_disk_space_kb_v2,
    group_containers_by_filesystem,
    run_code_in_con,
    wait_and_get_container,
    write_code_in_con,
)
from src.typing.config import AppConfig
from src.typing.training import (
    SpecialEnvironmentGetterCodeTrainData,
    StrategyCodeTrainData,
    StrategyListTrainData,
    save_train_data,
)

RUN_ID = generate_readable_run_id()
COMMIT_ID = get_formatted_repo_info()


def _build_metrics_collector(config: AppConfig) -> MetricsCollector:
    output_dir = (
        config.observability.metrics_output_path or config.train_data_save_folder
    )
    return MetricsCollector(
        run_id=RUN_ID,
        output_dir=output_dir,
        enabled=config.observability.enabled,
        record_inference=config.observability.record_inference,
        record_phases=config.observability.record_phases,
        record_resources=config.observability.record_resources,
    )


def _record_phase_metric(
    collector: MetricsCollector,
    phase_name: str,
    duration_ms: float,
    success: bool,
    episode_id: Optional[str] = None,
    retry_count: int = 0,
    error_message: Optional[str] = None,
) -> Optional[str]:
    collector.record_phase_safe(
        PhaseMetric(
            phase_name=phase_name,
            run_id=collector.run_id,
            episode_id=episode_id,
            duration_ms=duration_ms,
            success=success,
            retry_count=retry_count,
            error_message=error_message,
        )
    )
    return collector.flush()


def _run_backend_smoke_test(backend_session) -> None:
    smoke_test = getattr(backend_session, "smoke_test", None)
    if smoke_test is None:
        return

    logger.info("Running quick inference test...")
    test_text = smoke_test()
    logger.info(f"Inference test response: {test_text}")


def main(config_file: str = "./config/config-container.toml"):
    logger.info(f"Run ID: {RUN_ID}")
    logger.info(f"Commit ID: {COMMIT_ID}")

    with open(config_file, "r") as f:
        config_dict = toml.load(f)
    try:
        config = AppConfig(**unflatten_toml_dict(config_dict))
    except pydantic.ValidationError as e:
        logger.info(f"Config validation error: {e}")
        return

    metrics_collector = _build_metrics_collector(config)
    metrics_output_path: Optional[str] = None

    full_run_data = {
        "run_id": RUN_ID,
        "commit_id": COMMIT_ID,
        "config": config_file,
        "hardware_tags": list(config.observability.hardware_tags),
        "load_tags": list(config.observability.load_tags),
        "initial_env_info": {},
        "exploration": [],
        "strategy_generation": [],
        "strategy_execution": [],
        "metrics_summary": {},
        "metrics_path": None,
    }

    docker_client = docker.from_env()
    backend_stack = ExitStack()
    atexit.register(backend_stack.close)

    logger.info(f"Model name: {config.model_name}")
    backend_context = resolve_backend_context(config)
    if backend_context is None:
        raise RuntimeError(f"No backend context registered for {config.model_name}")

    # Create genner based on model configuration
    if config.model_name == "qwen-peft":
        # Use PEFT-based model
        backend_session = backend_stack.enter_context(backend_context)
        _run_backend_smoke_test(backend_session)
        genner = backend_session.genner
    elif config.model_name.startswith("llama:"):
        backend_session = backend_stack.enter_context(backend_context)
        _run_backend_smoke_test(backend_session)
        genner = backend_session.genner
    elif config.model_name.startswith("vllm"):
        backend_session = backend_stack.enter_context(backend_context)
        _run_backend_smoke_test(backend_session)
        genner = backend_session.genner
    elif config.model_name == "claude":
        # Claude API support
        backend_session = backend_stack.enter_context(backend_context)
        _run_backend_smoke_test(backend_session)
        genner = backend_session.genner

    else:
        # Use regular Ollama-based model
        backend_session = backend_stack.enter_context(backend_context)
        _run_backend_smoke_test(backend_session)
        genner = backend_session.genner

    genner = MetricsGenner(genner, metrics_collector)

    # Start containers dynamically if configured
    if config.dynamic_container:
        if not config.docker_compose_dir:
            logger.error("dynamic_container is true but docker_compose_dir is empty")
            return

        compose_dir = Path(config.docker_compose_dir)
        if not compose_dir.exists():
            logger.error(f"docker_compose_dir does not exist: {compose_dir}")
            return

        logger.info(f"Starting containers from {compose_dir}...")
        cmd = ["docker", "compose", "up", "-d", "--build"]
        logger.debug(f"Running command: {' '.join(cmd)} (cwd={compose_dir})")
        result = subprocess.run(
            cmd,
            cwd=str(compose_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            logger.error(
                f"Docker compose failed (returncode={result.returncode})\n"
                f"Command: {' '.join(cmd)}\n"
                f"Working dir: {compose_dir}\n"
                f"--- stdout ---\n{stdout or '<empty>'}\n"
                f"--- stderr ---\n{stderr or '<empty>'}"
            )
            if "error getting credentials" in stderr.lower():
                logger.error(
                    "Detected Docker credential helper issue. "
                    "In WSL, remove credsStore from ~/.docker/config.json (set it to {})."
                )
            return
        if result.stdout.strip():
            logger.debug(f"Docker compose stdout:\n{result.stdout}")
        logger.info("Containers started, waiting for them to be ready...")

    containers = []
    for container_id in config.container_ids:
        containers.append(wait_and_get_container(docker_client, container_id))

    env_info_dict: Dict[str, str] = {}
    containers_free_space: Dict[str, Tuple[Optional[float], float]] = {}
    for container in containers:
        assert container.id is not None

        rw_size_kb, free_space_v1_kb = get_container_free_disk_space_kb_v1(
            docker_client, container
        )

        if rw_size_kb is None or free_space_v1_kb is None:
            logger.warning(
                f"Failed to get read/write size or free space for container {container.id} using V1 method. "
            )

        free_space_v2_kb = get_container_free_disk_space_kb_v2(container)

        env_info = (
            f"Container ID: {container.id}\n"
            f"Read/Write Size: {rw_size_kb} KB\n"
            f"Free Disk Space (v1): {free_space_v1_kb} KB\n"
            f"Free Disk Space (v2): {free_space_v2_kb} KB\n"
        )
        logger.info(
            f"Env info for the container {container.id[:12]} is: \n{env_info.strip()}"
        )

        env_info_dict[container.id] = env_info
        containers_free_space[container.id] = (
            free_space_v1_kb,
            free_space_v2_kb,
        )

    full_run_data["initial_env_info"] = env_info_dict

    # Convert the dictionary values to a list
    env_infos: List[str] = list(env_info_dict.values())

    # Check if episode config exists to determine v1 vs v2
    if hasattr(config, "episode") and config.episode is not None:
        logger.info("Using NSL v2 episode-based execution")

        # Run v2 episode-based execution
        episode_started_at = time.perf_counter()
        try:
            episode_result = run_episode_v2(genner, docker_client, containers, config)
        except Exception as exc:
            metrics_output_path = (
                _record_phase_metric(
                    metrics_collector,
                    "episode_v2",
                    (time.perf_counter() - episode_started_at) * 1000,
                    False,
                    error_message=str(exc),
                )
                or metrics_output_path
            )
            raise
        metrics_output_path = (
            _record_phase_metric(
                metrics_collector,
                "episode_v2",
                (time.perf_counter() - episode_started_at) * 1000,
                episode_result["success"],
                episode_id=episode_result["episode_id"],
            )
            or metrics_output_path
        )

        # Extract results for compatibility with existing data_collector interface
        space_freed_kb = episode_result["space_freed_kb"]
        success = episode_result["success"]
        trajectory = episode_result["trajectory"]

        # Create v1-compatible training data structure from prompt/response pairs
        v2_train_data = []
        for prompt_resp in episode_result["prompt_responses"]:
            v2_train_data.append(
                {
                    "run_id": RUN_ID,
                    "version": COMMIT_ID,
                    "prompt": prompt_resp["prompt"],
                    "raw_response": prompt_resp["raw_response"],
                    "interaction_type": prompt_resp["interaction_type"],
                    "timestamp": prompt_resp["timestamp"],
                    "success": prompt_resp["success"],
                    "error_message": prompt_resp["error_message"],
                }
            )

        # Save v2 training data (maintain v1 interface for data_collector)
        save_train_data("episode_v2", v2_train_data, config.train_data_save_folder)

        # Update full_run_data for v2
        full_run_data["episode_execution"] = v2_train_data
        full_run_data["trajectory"] = trajectory
        full_run_data["space_freed_kb"] = space_freed_kb

        logger.info(f"Episode completed - Space freed: {space_freed_kb} KB")
        logger.info(f"Episode success: {success}")
        logger.info(f"Total v2 interactions: {len(v2_train_data)}")

    else:
        logger.info("Using NSL v1 fixed pipeline execution")

        # Fall back to v1 execution (existing code)
        exploration_started_at = time.perf_counter()
        sp_env_infos, sp_egc_train_data, sp_env_info_hashes = (
            special_environment_getter_code_flow(
                genner, docker_client, containers, config, env_infos
            )
        )
        metrics_output_path = (
            _record_phase_metric(
                metrics_collector,
                "special_environment_getter_code",
                (time.perf_counter() - exploration_started_at) * 1000,
                True,
                retry_count=max(0, len(sp_egc_train_data) - len(sp_env_infos)),
            )
            or metrics_output_path
        )
        save_train_data(
            "special_environment_getter_code",
            sp_egc_train_data,
            config.train_data_save_folder,
        )

        full_run_data["exploration"] = sp_egc_train_data

        logger.info(f"Special environment infos: \n{sp_env_infos}")
        logger.info(f"`len(sp_egc_train_data)`: {len(sp_egc_train_data)}")

        strategy_list_started_at = time.perf_counter()
        strategies, strategy_list_train_data, strategies_hash = strategy_list_flow(
            genner, config, env_infos, sp_env_infos, sp_env_info_hashes
        )
        metrics_output_path = (
            _record_phase_metric(
                metrics_collector,
                "strategy_list",
                (time.perf_counter() - strategy_list_started_at) * 1000,
                True,
                retry_count=max(0, len(strategy_list_train_data) - 1),
            )
            or metrics_output_path
        )
        save_train_data(
            "strategy_list", strategy_list_train_data, config.train_data_save_folder
        )

        full_run_data["strategy_generation"] = strategy_list_train_data
        logger.info(f"Strategy list: \n{strategies}")
        logger.info(f"`len(strategy_list_train_data)`: {len(strategy_list_train_data)}")

        if not strategies:
            logger.warning("No strategies were generated. Skipping execution.")
            space_freed_kb = 0.0
        else:
            strategy_to_run = random.choice(strategies)
            logger.info(f"Executing ONE random strategy: {strategy_to_run}")

            strategy_code_started_at = time.perf_counter()
            strat_code, strat_code_hash, strat_code_train_data, space_freed_kb = (
                strategy_code_flow(
                    genner,
                    docker_client,
                    containers,
                    containers_free_space,
                    config,
                    strategy_to_run,
                    strategies_hash,
                    env_info_dict,
                    sp_env_infos,
                    sp_env_info_hashes,
                )
            )
            metrics_output_path = (
                _record_phase_metric(
                    metrics_collector,
                    "strategy_code",
                    (time.perf_counter() - strategy_code_started_at) * 1000,
                    True,
                    retry_count=max(0, len(strat_code_train_data) - 1),
                )
                or metrics_output_path
            )

            strategy_execution_data = {
                "strategy_text": strategy_to_run,
                "all_code_attempts": strat_code_train_data,
                "final_successful_output": strat_code,
                "all_metrics_for_this_strategy": {
                    "space_freed_kb": space_freed_kb,
                    "final_code_hash": strat_code_hash,
                    "total_attempts": len(strat_code_train_data),
                },
            }

            save_train_data(
                "strategy_code", strat_code_train_data, config.train_data_save_folder
            )

            full_run_data["strategy_execution"].append(strategy_execution_data)
            logger.info(f"Strategy code: \n{strat_code}")
            print(f"Space freed: {space_freed_kb} KB")

    metrics_output_path = metrics_collector.flush() or metrics_output_path
    full_run_data["metrics_summary"] = metrics_collector.summary()
    full_run_data["metrics_path"] = metrics_output_path

    output_json_path = Path(config.train_data_save_folder) / "LAST_RUN_LOG.json"

    try:
        with open(output_json_path, "w") as f:
            json.dump(full_run_data, f, indent=4, default=str)
        logger.info(f"Raw agent log saved to {output_json_path}")
    except Exception as e:
        logger.error(f"Failed to save raw agent log: {e}")

    logger.info("Main function finished.")


def special_environment_getter_code_flow(
    genner: Genner,
    docker_client: docker.DockerClient,
    containers: List[DockerContainer],
    config: AppConfig,
    env_infos: List[str],
):
    # Output variables
    sp_env_infos: List[str] = []
    sp_egc_train_data: List[SpecialEnvironmentGetterCodeTrainData] = []
    sp_env_info_hashes: List[str] = []

    # Loop variables
    current_attempt = 0
    regen_count = 0
    should_regen = False
    had_succeed = False
    error_sources: List[str] = []
    error_contexts: List[str] = []
    latest_generation: Optional[str] = None

    while len(sp_env_infos) < config.special_egc.count:
        if had_succeed:
            had_succeed = False
            error_sources: List[str] = []
            error_contexts: List[str] = []
            latest_generation: Optional[str] = None
            current_attempt = 0
            regen_count = 0

        if current_attempt > config.special_egc.max_retries:
            raise Exception(
                "Special environment getter code generation failed. Max retries exceeded, crashing on purpose."
            )

        logger.debug(f"Current attempt: {current_attempt}")
        current_attempt += 1

        if should_regen:
            logger.info(f"Regenning after {len(error_contexts)} mistakes for SP EGC...")

            should_regen = False
            assert latest_generation is not None

            match regenerate_env_discovery_code(
                genner,
                regen_count,
                [error_sources[-1]] if error_sources else [],  # Only last error source
                [error_contexts[-1]] if error_contexts else [],
                latest_generation,
            ):
                case Ok((raw_response, prompt)):
                    logger.info(f"Regenerated a new code: \n{raw_response}")
                    pass
                case Err(error_message):
                    logger.error(f"Failed to regenerate code: \n{error_message}")
                    continue

            regen_count += 1
        else:
            logger.info("Generating new code for SP EGC...")
            match generate_special_environment_getter_code(
                genner, env_infos, sp_env_infos
            ):
                case Ok((raw_response, prompt)):
                    logger.info(f"Regenerated a new code: \n{raw_response}")
                    pass
                case Err(error_message):
                    logger.error(f"Failed to generate code: \n{error_message}")
                    continue
        logger.debug(f"Raw response generated: \n{raw_response}")

        logger.trace("Extracting code from raw response...")
        match genner.extract_code(raw_response):
            case Ok(code):
                pass
            case Err(error_message):
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_err": error_message,
                    }
                )
                logger.error(f"Failed to extract code. Error: \n{error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_extraction")

                continue

        logger.trace("Writing code in container...")
        in_container_code_path, code = write_code_in_con(
            docker_client,
            containers[config.main_container_idx],
            host_cache_folder=Path(config.code_host_cache_path) / "special_egc",
            code=code,
            postfix="",
            in_container_path="/",
        )
        logger.trace(
            f"Code written in container successfully with the name of {in_container_code_path}"
        )

        logger.trace("Validating code offline...")
        match validate_code_offline(code):
            case Ok(_):
                pass
            case Err(error_message):
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_ok": code,
                        "validation_result_if_err": error_message,
                    }
                )

                logger.error(f"Failed to validate code. Error: \n{error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_validation")

                continue

        logger.trace("Running code in container...")
        match run_code_in_con(
            containers[config.main_container_idx],
            in_container_code_path,
        ):
            case Ok(execution_output):
                if execution_output.strip() == "":
                    logger.error("Execution output is empty")

                    sp_egc_train_data.append(
                        {
                            "run_id": RUN_ID,
                            "version": COMMIT_ID,
                            "prompt": prompt,
                            "raw_response": raw_response,
                            "current_attempt": current_attempt,
                            "max_attempts": config.special_egc.max_retries,
                            "extract_result_if_ok": code,
                            "run_result_if_empty": "<nothing>",
                        }
                    )

                    should_regen = True
                    latest_generation = raw_response
                    error_contexts.append(
                        "Execution output is empty, please check the code."
                    )
                    error_sources.append("code_run")

                    continue

                execution_hash = string_hash(execution_output)
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_ok": code,
                        "run_result_if_ok": execution_output,
                        "run_result_if_ok_hash": execution_hash,
                    }
                )
                sp_env_infos.append(execution_output)
                sp_env_info_hashes.append(execution_hash)
                had_succeed = True

                logger.info(
                    f"Successfully ran code in container. Output: \n{execution_output.strip()}"
                )

            case Err(error_message):
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_ok": code,
                        "run_result_if_err": error_message,
                    }
                )

                logger.error(
                    f"Failed to run code in container. Error: \n{error_message}"
                )

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_run")

                continue

    logger.info("Special EGC stage completed.")
    logger.info(f"`len(sp_egc_train_data)`: {len(sp_egc_train_data)}")
    logger.info(f"`len(sp_env_infos)`: {len(sp_env_infos)}")

    for i, env_info in enumerate(sp_env_infos):
        logger.info(
            f"Special environment info {i + 1}: \n{env_info}\nHash: {sp_env_info_hashes[i]}"
        )

    return sp_env_infos, sp_egc_train_data, sp_env_info_hashes


def strategy_list_flow(
    genner: Genner,
    config: AppConfig,
    env_infos: List[str],
    special_env_infos: List[str],
    special_env_info_hashes: List[str],
):
    # Output variables
    strategies: Optional[List[str]] = None
    strategies_hash: Optional[str] = None
    strategy_list_train_data: List[StrategyListTrainData] = []

    # Loop variables
    current_attempt = 0
    regen_count = 0
    should_regen = False
    error_contexts: List[str] = []
    latest_generation: Optional[str] = None

    while strategies is None:
        if current_attempt > config.strategy_list.max_retries:
            raise Exception(
                "Strategy list generation failed. Max retries exceeded, crashing on purpose."
            )

        logger.debug(f"Current try: {current_attempt}")
        current_attempt += 1

        if should_regen:
            should_regen = False
            assert latest_generation is not None

            match regenerate_list(
                genner, regen_count, error_contexts, latest_generation
            ):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to regenerate list: {error_message}")
                    continue

            regen_count += 1
        else:
            match generate_strategy_list(genner, env_infos, special_env_infos, []):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to generate code: {error_message}")
                    continue

        match genner.extract_list(raw_response):
            case Ok(new_strategies):
                strategies = new_strategies
                strategies_hash = string_hash(",".join(strategies))

                strategy_list_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "special_env_info_hashes": special_env_info_hashes,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_list.max_retries,
                        "extract_result_if_ok": new_strategies,
                        "extract_result_if_ok_hash": strategies_hash,
                    }
                )
            case Err(error_message):
                strategy_list_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "special_env_info_hashes": special_env_info_hashes,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_list.max_retries,
                        "extract_result_if_err": error_message,
                    }
                )
                logger.error(f"Failed to extract list: \n{error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)

                continue

    assert strategies is not None
    assert strategies_hash is not None

    logger.info("Strategy list stage completed.")
    logger.info(f"Strategy list training data: {strategy_list_train_data}")
    logger.info(f"Strategies generated: {strategies}")

    return strategies, strategy_list_train_data, strategies_hash


def strategy_code_flow(
    genner: Genner,
    docker_client: docker.DockerClient,
    containers: List[DockerContainer],
    containers_free_space: Dict[str, Tuple[Optional[float], float]],
    config: AppConfig,
    strategy: str,
    strategies_hash: str,
    env_info_dict: Dict[str, str],
    special_env_infos: List[str],
    special_env_info_hashes: List[str],
):
    # Output
    strat_code: Optional[str] = None
    strat_code_hash: Optional[str] = None
    strat_code_train_data: List[StrategyCodeTrainData] = []
    strat_space_freed_kb = 0

    # Loop variables
    current_attempt = 0
    regen_count = 0
    should_regen = False
    error_sources: List[str] = []
    error_contexts: List[str] = []
    latest_generation: Optional[str] = None
    filesystem_groups = group_containers_by_filesystem(docker_client, containers)

    logger.info(
        f"Detected {len(filesystem_groups)} unique backing filesystems "
        f"across {len(containers)} containers for v1 strategy execution"
    )

    while strat_code is None:
        if current_attempt > config.strategy_code.max_retries:
            logger.error(
                "Strategy code generation failed. Max retries exceeded, crashing on purpose."
            )
            raise Exception("Strategy code generation failed. Max retries exceeded.")

        logger.debug(f"Cur retry: {current_attempt}")
        current_attempt += 1

        if should_regen:
            should_regen = False
            assert latest_generation is not None

            match regenerate_code(
                genner, regen_count, error_sources, error_contexts, latest_generation
            ):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to regenerate code: {error_message}")
                    continue

            regen_count += 1
        else:
            match generate_strategy_code(
                genner, strategy, list(env_info_dict.values()), special_env_infos
            ):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to generate code: {error_message}")
                    continue

        match genner.extract_code(raw_response):
            case Ok(code):
                pass
            case Err(error_message):
                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "extract_result_if_err": error_message,
                    }
                )
                logger.error(f"Failed to extract code: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_extraction")

                continue

        in_container_code_path, code = write_code_in_con(
            docker_client,
            containers[config.main_container_idx],
            host_cache_folder=Path(config.code_host_cache_path) / "strategy_code",
            code=code,
            postfix="",
            in_container_path="/",
        )

        match validate_code_offline(code):
            case Ok(_):
                pass
            case Err(error_message):
                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "validation_result_if_err": error_message,
                    }
                )

                logger.error(f"Failed to validate code: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_validation")

                continue

        match run_code_in_con(
            containers[config.main_container_idx],
            in_container_code_path,
        ):
            case Ok(execution_output):
                # Check if there's any space freed in V1 or V2
                v2_space_measurements: Dict[str, Tuple[float, float]] = {}
                for container in containers:
                    assert container.id is not None

                    # Get old space measurements
                    old_free_space_v1_kb, old_free_space_v2_kb = containers_free_space[
                        container.id
                    ]

                    # DETAILED DEBUG: Show what we're comparing
                    logger.info(f"SPACE DEBUG for container {container.id[:12]}:")
                    logger.info(f"  BEFORE execution:")
                    logger.info(f"  V1 (Docker): {old_free_space_v1_kb} KB")
                    logger.info(f"  V2 (df -k):  {old_free_space_v2_kb} KB")

                    # Check if files exist before measuring new space
                    file_check_cmd = (
                        "ls -la /tmp/big_cleanup/ 2>/dev/null | wc -l || echo '0'"
                    )
                    try:
                        file_count_result = container.exec_run(file_check_cmd)
                        file_count = file_count_result.output.decode().strip()
                        logger.info(
                            f"Files in /tmp/big_cleanup after execution: {file_count}"
                        )
                    except Exception as e:
                        logger.info(f"Could not check files: {e}")

                    # Get new space measurements
                    new_rw_size_kb, new_free_space_v1_kb = (
                        get_container_free_disk_space_kb_v1(docker_client, container)
                    )
                    new_free_space_v2_kb = get_container_free_disk_space_kb_v2(
                        container
                    )

                    logger.info(f"AFTER execution:")
                    logger.info(f"  V1 (Docker): {new_free_space_v1_kb} KB")
                    logger.info(f"  V2 (df -k):  {new_free_space_v2_kb} KB")

                    # Calculate differences
                    space_freed_v1 = (
                        (new_free_space_v1_kb - old_free_space_v1_kb)
                        if old_free_space_v1_kb is not None
                        and new_free_space_v1_kb is not None
                        else None
                    )
                    space_freed_v2 = new_free_space_v2_kb - old_free_space_v2_kb

                    logger.info(f"SPACE CHANGES:")
                    logger.info(f"  V1 change: {space_freed_v1} KB")
                    logger.info(f"  V2 change: {space_freed_v2} KB")

                    v2_space_measurements[container.id] = (
                        old_free_space_v2_kb,
                        new_free_space_v2_kb,
                    )

                    env_info_dict[container.id] = (
                        f"Container ID: {container.id}\n"
                        f"Read/Write Size: {new_rw_size_kb} KB\n"
                        f"Free Disk Space (v1): {new_free_space_v1_kb} KB\n"
                        f"Free Disk Space (v2): {new_free_space_v2_kb} KB\n"
                    )
                    containers_free_space[container.id] = (
                        new_free_space_v1_kb,
                        new_free_space_v2_kb,
                    )

                containers_space_freed_kb = calculate_deduplicated_space_freed(
                    v2_space_measurements,
                    filesystem_groups,
                )
                logger.info(
                    f"Deduplicated v1 space change across filesystems: "
                    f"{containers_space_freed_kb} KB"
                )

                strat_code = code
                strat_code_hash = string_hash(code)
                strat_space_freed_kb = containers_space_freed_kb

                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "extract_result_if_ok": code,
                        "run_result_if_ok": execution_output,
                        "run_result_if_ok_hash": strat_code_hash,
                        "this_code_space_change_kb": containers_space_freed_kb,
                    }
                )

                break
            case Err(error_message):
                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "extract_result_if_ok": code,
                        "run_result_if_err": error_message,
                    }
                )

                logger.error(f"Failed to run code in container: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_run")

    logger.info(f"Strategy code completed on strat \n{strategy}.")
    logger.info(f"Strategy code train data: {strat_code_train_data}")
    logger.info(f"Strategy code: {strat_code}")
    logger.info(f"Total space freed: {strat_space_freed_kb} KB")

    return (
        strat_code,
        strat_code_hash,
        strat_code_train_data,
        strat_space_freed_kb,
    )


def run_episode_v2(
    genner: Genner,
    docker_client: docker.DockerClient,
    containers: List[DockerContainer],
    config: AppConfig,
    run_id: Optional[str] = None,
    commit_id: Optional[str] = None,
    save_incremental: bool = True,
) -> Dict[str, Any]:
    """Unified mode system with orchestrator → mode delegation."""

    # Initialize ModeController (replaces AgentV2)
    from src.mode_controller import ModeController

    mode_controller = ModeController(genner, config)
    mode_controller.initialize_docker(
        docker_client=docker_client,
        container=containers[config.main_container_idx],
        host_cache_folder=Path(config.code_host_cache_path) / "episode_v2_mode",
    )

    run_id = run_id or RUN_ID
    commit_id = commit_id or COMMIT_ID

    # Start episode
    episode_id = f"ep_{run_id}_{int(time.time())}"
    mode_controller.start_episode(episode_id)

    # Measure initial disk space (preserve existing measurement logic)
    filesystem_groups = group_containers_by_filesystem(docker_client, containers)
    logger.info(
        f"Detected {len(filesystem_groups)} unique backing filesystems "
        f"across {len(containers)} containers for v2 episode execution"
    )

    initial_free_space: Dict[str, float] = {}
    for container in containers:
        assert container.id is not None
        initial_free_space[container.id] = get_container_free_disk_space_kb_v2(
            container
        )

    # Collect all prompt/response pairs for training data
    all_prompt_responses = []
    partial = False
    error_message: Optional[str] = None

    logger.info(f"Starting episode {episode_id} with unified mode system")
    logger.info(f"Orchestrator budget: {mode_controller.budget.total_budget} decisions")

    # Unified mode execution loop
    while mode_controller.can_take_action():
        step = mode_controller.episode_state.current_step
        budget_remaining = mode_controller.budget.remaining

        logger.info(
            f"=== Step {step} - Orchestrator Budget: {budget_remaining}/{mode_controller.budget.total_budget} ==="
        )

        try:
            # 1. Orchestrator makes strategic decision
            orchestrator_decision = mode_controller.execute_orchestrator_mode()

            logger.info(f"Orchestrator → {orchestrator_decision.target_mode.value}")
            logger.info(f"Instruction: {orchestrator_decision.instruction}")
            logger.info(f"Reasoning: {orchestrator_decision.reasoning}")

            # Record orchestrator interaction
            from datetime import datetime

            orchestrator_prompt_response = {
                "prompt": f"Orchestrator planning (budget: {budget_remaining})",
                "raw_response": orchestrator_decision.raw_response,
                "timestamp": datetime.now().isoformat(),
                "interaction_type": "orchestrator",
                "success": True,
                "error_message": None,
            }
            all_prompt_responses.append(orchestrator_prompt_response)

            # 2. Execute delegated mode
            mode_result = mode_controller.execute_mode(
                orchestrator_decision.target_mode, orchestrator_decision.instruction
            )

            logger.info(
                f"{orchestrator_decision.target_mode.value} → {'SUCCESS' if mode_result.success else 'FAILED'}"
            )
            logger.info(f"Result: {mode_result.content[:200]}...")

            # Record mode interaction for training data
            if mode_result.prompt_response_pair:
                mode_prompt_response = {
                    "prompt": mode_result.prompt_response_pair.prompt,
                    "raw_response": mode_result.prompt_response_pair.raw_response,
                    "timestamp": mode_result.prompt_response_pair.timestamp,
                    "interaction_type": mode_result.prompt_response_pair.interaction_type,
                    "success": mode_result.prompt_response_pair.success,
                    "error_message": mode_result.prompt_response_pair.error_message,
                }
                all_prompt_responses.append(mode_prompt_response)

                # Save training data incrementally after each step
                current_step_data = []
                for prompt_resp in all_prompt_responses:
                    current_step_data.append(
                        {
                            "run_id": run_id,
                            "version": commit_id,
                            "prompt": prompt_resp["prompt"],
                            "raw_response": prompt_resp["raw_response"],
                            "interaction_type": prompt_resp["interaction_type"],
                            "timestamp": prompt_resp["timestamp"],
                            "success": prompt_resp["success"],
                            "error_message": prompt_resp["error_message"],
                        }
                    )

                # Save incremental data
                if save_incremental:
                    save_train_data(
                        f"episode_v2_step_{len(mode_controller.episode_state.mode_history)}",
                        current_step_data,
                        config.train_data_save_folder,
                    )

            # 3. Update episode state
            mode_controller.episode_state.add_mode_execution(
                orchestrator_decision.target_mode,
                orchestrator_decision.instruction,
                mode_result,
                orchestrator_duration_ms=orchestrator_decision.duration_ms,
            )

            # Handle failures
            if not mode_result.success:
                logger.warning(
                    f"{orchestrator_decision.target_mode.value} failed: {mode_result.error_message}"
                )

        except Exception as e:
            logger.error(f"Mode execution error: {e}")
            # Add error to scratchpad
            mode_controller.scratchpad.append(f"Episode failed: {e}")
            partial = True
            error_message = str(e)
            break

    # Measure final disk space (preserve existing measurement logic)
    final_free_space: Dict[str, float] = {}
    space_measurements: Dict[str, Tuple[float, float]] = {}
    measurement_errors: List[str] = []

    for i, container in enumerate(containers):
        assert container.id is not None
        try:
            final_free_space[container.id] = get_container_free_disk_space_kb_v2(container)
            space_freed = final_free_space[container.id] - initial_free_space[container.id]
            space_measurements[container.id] = (
                initial_free_space[container.id],
                final_free_space[container.id],
            )
            logger.info(f"Container {i}: {space_freed:.2f} KB freed")
        except Exception as e:
            logger.error(f"Container {i} final measurement failed: {e}")
            measurement_errors.append(container.id)

    total_space_freed = calculate_deduplicated_space_freed(
        space_measurements,
        filesystem_groups,
    )

    logger.info(f"Total space freed: {total_space_freed:.2f} KB")
    logger.info(f"Episode completed - Success: {total_space_freed > 0.0}")

    # Update episode state with final results
    mode_controller.episode_state.total_space_freed = total_space_freed

    # Get complete episode summary for trajectory
    episode_summary = mode_controller.get_episode_summary()

    return {
        "trajectory": episode_summary,  # Complete mode execution history
        "prompt_responses": all_prompt_responses,  # All orchestrator + mode conversations
        "space_freed_kb": total_space_freed,
        "success": total_space_freed > 0.0,
        "episode_runtime_success": total_space_freed > 0.0,
        "episode_id": episode_id,
        "action_count": len(mode_controller.episode_state.mode_history),
        "partial": partial,
        "error_message": error_message,
        "space_measurements": space_measurements,
        "filesystem_groups": [group.__dict__ for group in filesystem_groups],
        "measurement_errors": measurement_errors,
    }


if __name__ == "__main__":
    import sys

    config_file_path = "./config/config-container.toml"
    if len(sys.argv) > 1:
        if sys.argv[1].endswith(".toml"):
            config_file_path = sys.argv[1]

    main(config_file=config_file_path)
