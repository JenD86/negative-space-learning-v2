import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse

from loguru import logger
from openai import OpenAI

from src.backend.utils import get_recent_log_lines, is_http_ready, terminate_process
from src.genner import get_genner
from src.genner.config import VllmConfig
from src.typing.config import AppConfig

from .session import BackendSession


def _is_gguf_model(model: str) -> bool:
    return "gguf" in model.lower() or model.lower().endswith(".gguf")


def _normalize_vllm_model(model: str) -> str:
    normalized_model = model.strip()
    legacy_model_aliases = {
        "qwen2.5-coder:7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5:7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    }
    return legacy_model_aliases.get(normalized_model, normalized_model)


def _build_vllm_config(app_config: AppConfig, endpoint: str, timeout: int) -> VllmConfig:
    config = VllmConfig()
    if app_config.model_name.startswith("vllm:"):
        config.model = app_config.model_name.split(":", 1)[1].strip()

    resolved_model = _normalize_vllm_model(config.model)
    if resolved_model != config.model.strip():
        logger.info(f"Resolved model alias '{config.model}' to '{resolved_model}'")
    config.model = resolved_model
    config.backend = "vllm"
    config.endpoint = endpoint
    config.timeout = timeout
    config.gpu_memory_utilization = app_config.gpu_memory_utilization
    config.temperature = app_config.inference.temperature
    config.max_tokens = app_config.inference.max_tokens
    return config


def _build_vllm_command(
    model: str,
    host: str,
    port: int,
    gpu_memory_utilization: float,
) -> list[str]:
    vllm_bin = shutil.which("vllm")
    gpu_memory_utilization_arg = str(gpu_memory_utilization)
    extra_args: list[str] = []
    if _is_gguf_model(model):
        extra_args.extend(["--quantization", "gguf"])
    if vllm_bin:
        return [
            vllm_bin,
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            gpu_memory_utilization_arg,
            *extra_args,
        ]
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        gpu_memory_utilization_arg,
        *extra_args,
    ]


def _wait_for_vllm_ready(
    models_url: str,
    process: subprocess.Popen,
    timeout_s: int,
    command: list[str],
    stderr_log_path: Optional[str] = None,
) -> None:
    deadline = time.time() + timeout_s
    last_output_size = 0
    last_output_change_time = time.time()
    freeze_threshold = 60

    while time.time() < deadline:
        if is_http_ready(models_url):
            return

        if process.poll() is not None:
            logs = ""
            if stderr_log_path:
                try:
                    logs = Path(stderr_log_path).read_text()[-2000:]
                except Exception:
                    pass
            raise RuntimeError(
                "vLLM process exited before becoming ready "
                f"(code={process.returncode}, command={' '.join(command)}). Last logs:\n{logs}"
            )

        if stderr_log_path:
            try:
                current_size = Path(stderr_log_path).stat().st_size
                if current_size != last_output_size:
                    last_output_size = current_size
                    last_output_change_time = time.time()
                else:
                    seconds_silent = time.time() - last_output_change_time
                    if seconds_silent > freeze_threshold:
                        recent_lines = get_recent_log_lines(stderr_log_path, n=15)
                        logger.warning(
                            f"vLLM appears frozen (no output change for {seconds_silent:.0f}s). "
                            f"Recent log:\n{recent_lines}"
                        )
            except FileNotFoundError:
                pass

        time.sleep(1)

    raise TimeoutError(
        f"Timed out waiting for vLLM readiness at {models_url} after {timeout_s}s"
    )


def _build_vllm_smoke_test(client: OpenAI, config: VllmConfig):
    def run_smoke_test() -> str:
        test_response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Who are you?"}],
            max_tokens=50,
            temperature=0.5,
        )
        return test_response.choices[0].message.content

    return run_smoke_test


def _build_vllm_session(
    client: OpenAI,
    config: VllmConfig,
    base_url: str,
    models_url: str,
    stderr_log_path: Optional[str] = None,
    process: Optional[subprocess.Popen] = None,
) -> BackendSession:
    genner = get_genner("vllm", vllm_config=config, oai_client=client)
    return BackendSession(
        genner=genner,
        smoke_test=_build_vllm_smoke_test(client, config),
        client=client,
        config=config,
        base_url=base_url,
        models_url=models_url,
        stderr_log_path=stderr_log_path,
        process=process,
    )


@contextmanager
def setup_vllm(
    app_config: AppConfig,
    *,
    endpoint: str = "http://localhost:8000",
    timeout: int = 500,
) -> Iterator[BackendSession]:
    config = _build_vllm_config(app_config, endpoint=endpoint, timeout=timeout)

    endpoint = config.endpoint.rstrip("/")
    base_url = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
    models_url = f"{base_url}/models"
    api_key = config.api_key or "dummy"
    client = OpenAI(api_key=api_key, base_url=base_url)

    if is_http_ready(models_url):
        logger.info(f"Using existing server at {base_url}")
        yield _build_vllm_session(client, config, base_url, models_url)
        return

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000

    logger.info(
        f"No server detected at {base_url}. Starting vllm backend for {config.model} on port {port}..."
    )

    command = _build_vllm_command(
        config.model,
        host,
        port,
        config.gpu_memory_utilization,
    )

    logger.info(f"vLLM command: {' '.join(command)}")

    stderr_log = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"vllm_stderr_{config.model.replace('/', '_')}_",
        suffix=".log",
        delete=False,
    )
    stderr_log_path = stderr_log.name
    stderr_log.close()
    logger.info(f"vLLM stderr log: {stderr_log_path}")

    process = None
    try:
        with open(stderr_log_path, "a") as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )

        startup_timeout = max(config.timeout, timeout)
        _wait_for_vllm_ready(
            models_url,
            process,
            startup_timeout,
            command,
            stderr_log_path,
        )

        logger.info(f"vLLM server is ready at {base_url}")
        yield _build_vllm_session(
            client,
            config,
            base_url,
            models_url,
            stderr_log_path=stderr_log_path,
            process=process,
        )
    finally:
        if process is not None:
            terminate_process(process, "vLLM")


__all__ = ["setup_vllm"]
