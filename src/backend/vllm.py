import importlib
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from docker import DockerClient
from docker.errors import NotFound
from loguru import logger
from openai import OpenAI

from src.backend.utils import is_http_ready
from src.genner import get_genner
from src.genner.config import VllmConfig
from src.typing.config import AppConfig

from .session import BackendSession

VLLM_IMAGE = "vllm/vllm-openai:latest"
CONTAINER_NAME_PREFIX = "nsl-vllm"
HF_CACHE_HOST = Path.home() / ".cache" / "huggingface"
HF_CACHE_CONTAINER = "/root/.cache/huggingface"
LOCAL_MODEL_CONTAINER_PATH = "/models/local"
LOCAL_ADAPTER_CONTAINER_PATH = "/adapters/local"
VLLM_NETWORK_MODE_ENV = "NSL_VLLM_NETWORK_MODE"
LEGACY_VLLM_MODEL_ALIASES = {
    "qwen2.5-coder:7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5:7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}


def _resolve_model_for_container(
    model: str, local_model_path: str | None
) -> tuple[str, str | None, bool]:
    """Returns (model_arg_for_vllm, host_mount_path, needs_bnb_flags)."""
    if local_model_path:
        return (
            LOCAL_MODEL_CONTAINER_PATH,
            str(Path(local_model_path).expanduser().resolve()),
            False,
        )
    lower = model.lower()
    return model, None, ("bnb-4bit" in lower or "bnb-8bit" in lower)


def _get_host_ip() -> str:
    """Get a routable host IP for container-to-host connectivity."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as exc:
        logger.debug(f"Failed to detect non-loopback host IP: {exc}")
        return "127.0.0.1"


def _get_network_mode() -> str:
    """Return the preferred vLLM network mode.

    Supported values:
    - auto: try the configured endpoint first, then a detected host IP
    - loopback: only use the configured endpoint
    - hostip: prefer the detected host IP endpoint
    """

    mode = os.getenv(VLLM_NETWORK_MODE_ENV, "auto").strip().lower()
    if mode in {"auto", "loopback", "hostip"}:
        return mode

    logger.warning(
        f"Invalid {VLLM_NETWORK_MODE_ENV} value '{mode}'; falling back to 'auto'"
    )
    return "auto"


def _build_models_urls(
    models_url: str,
    host_ip: str | None,
    port: int,
    network_mode: str,
    *,
    scheme: str,
) -> list[str]:
    if network_mode == "loopback":
        return [models_url]

    if host_ip in (None, "127.0.0.1"):
        if network_mode == "hostip":
            logger.warning(
                "Requested host IP networking, but no non-loopback host IP was "
                "detected. Falling back to the configured endpoint."
            )
        return [models_url]

    host_models_url = f"{scheme}://{host_ip}:{port}/v1/models"
    if network_mode == "hostip":
        return [host_models_url]

    if host_models_url == models_url:
        return [models_url]

    return [models_url, host_models_url]


def _read_gpu_memory_info_mb(device_index: int = 0) -> tuple[float, float] | None:
    pynvml = None
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (
            float(memory_info.free / (1024 * 1024)),
            float(memory_info.total / (1024 * 1024)),
        )
    except Exception:
        pass
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None

    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 2:
        return None

    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def wait_for_gpu_memory_release(
    *,
    min_free_memory_fraction: float,
    timeout_s: int,
    poll_interval_s: float = 2.0,
    device_index: int = 0,
) -> None:
    if timeout_s <= 0:
        return

    memory_info = _read_gpu_memory_info_mb(device_index)
    if memory_info is None:
        logger.warning(
            "Skipping GPU memory wait because GPU memory metrics are unavailable"
        )
        return

    deadline = time.monotonic() + timeout_s
    required_free_mb = memory_info[1] * min_free_memory_fraction

    while True:
        free_mb, total_mb = memory_info
        if free_mb >= required_free_mb:
            logger.info(
                f"GPU memory gate passed with {free_mb:.0f} MB free out of {total_mb:.0f} MB"
            )
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for GPU memory to be released "
                f"({free_mb:.0f} MB free, needed {required_free_mb:.0f} MB)"
            )

        time.sleep(poll_interval_s)
        memory_info = _read_gpu_memory_info_mb(device_index)
        if memory_info is None:
            logger.warning(
                "GPU memory metrics became unavailable while waiting; continuing without a wait gate"
            )
            return


def _build_vllm_config(
    app_config: AppConfig, endpoint: str, timeout: int
) -> VllmConfig:
    config = VllmConfig()
    if app_config.model_name.startswith("vllm:"):
        config.model = app_config.model_name.split(":", 1)[1].strip()

    normalized_model = config.model.strip()
    resolved_model = LEGACY_VLLM_MODEL_ALIASES.get(normalized_model, normalized_model)
    if resolved_model != normalized_model:
        logger.info(f"Resolved model alias '{normalized_model}' to '{resolved_model}'")
    config.model = resolved_model
    config.endpoint = endpoint
    config.timeout = timeout
    config.gpu_memory_utilization = app_config.gpu_memory_utilization
    config.temperature = app_config.inference.temperature
    config.max_tokens = app_config.inference.max_tokens
    if app_config.vllm is not None and app_config.vllm.chat_template_path:
        chat_template_path = Path(app_config.vllm.chat_template_path)
        config.chat_template = chat_template_path.read_text(encoding="utf-8")
    if app_config.vllm is not None:
        if app_config.vllm.served_model_name:
            config.model = app_config.vllm.served_model_name
        elif app_config.vllm.lora_adapter_path:
            config.model = "adapter"
    return config


def _build_container_command(
    model: str,
    gpu_memory_utilization: float,
    chat_template: str | None = None,
    lora_adapter_path: str | None = None,
    served_model_name: str | None = None,
    needs_bnb: bool | None = None,
) -> list[str]:
    cmd = [
        "--model",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    use_bnb = (
        needs_bnb
        if needs_bnb is not None
        else ("bnb-4bit" in model.lower() or "bnb-8bit" in model.lower())
    )
    if use_bnb:
        cmd.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    if lora_adapter_path:
        cmd.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"adapter={LOCAL_ADAPTER_CONTAINER_PATH}",
                "--max-lora-rank",
                "64",
            ]
        )
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    return cmd


def _container_name(port: int) -> str:
    return f"{CONTAINER_NAME_PREFIX}-{port}"


def _remove_stale_container(docker_client: DockerClient, name: str) -> None:
    try:
        stale = docker_client.containers.get(name)
        logger.info(f"Removing stale container '{name}'...")
        stale.remove(force=True)
    except NotFound:
        pass


def _start_vllm_container(
    name: str,
    port: int,
    vllm_command: list[str],
    extra_volumes: list[tuple[str, str]] | None = None,
) -> None:
    """Start a vLLM container using the docker CLI.

    We use subprocess instead of the Python SDK because the CLI supports the
    GPU device configuration used by the local environment.
    """
    HF_CACHE_HOST.mkdir(parents=True, exist_ok=True)

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--device",
        "nvidia.com/gpu=all",
        "--ipc=host",
        "--network=host",
        "-v",
        f"{HF_CACHE_HOST}:{HF_CACHE_CONTAINER}",
    ]
    for host_path, container_path in extra_volumes or []:
        docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
    docker_cmd.extend([VLLM_IMAGE, *vllm_command])

    logger.info(f"Docker command: {' '.join(docker_cmd)}")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start vLLM container: {result.stderr.strip()}")


def _wait_for_vllm_ready(
    models_urls: str | list[str],
    container,
    timeout_s: int,
    primary_timeout_s: int = 60,
) -> str:
    """Wait for vLLM to become ready, trying multiple endpoints.

    Returns the first URL that becomes ready.

    Args:
        models_urls: Candidate URLs to try in order
        container: Docker container handle
        timeout_s: Total timeout for all endpoints
        primary_timeout_s: Timeout for the first endpoint before trying alternates
    """
    if isinstance(models_urls, str):
        models_urls = [models_urls]

    deadline = time.time() + timeout_s
    start_time = time.time()
    next_log_at = 30.0

    primary_url = models_urls[0]
    alternate_urls = models_urls[1:]
    primary_deadline = time.time() + min(primary_timeout_s, timeout_s)
    last_checked_url = primary_url
    using_alternates = False
    while time.time() < deadline:
        if not using_alternates and alternate_urls and time.time() >= primary_deadline:
            using_alternates = True
            logger.info(
                "Primary endpoint unreachable, switching to alternate: "
                f"{alternate_urls[0]}"
            )

        candidate_urls = alternate_urls if using_alternates else [primary_url]
        if not candidate_urls:
            candidate_urls = [primary_url]

        for current_url in candidate_urls:
            last_checked_url = current_url
            if is_http_ready(current_url):
                return current_url

        container.reload()
        if container.status in ("exited", "dead"):
            logs = container.logs(tail=50).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM container exited unexpectedly (status={container.status}). "
                f"Logs:\n{logs}"
            )

        elapsed = time.time() - start_time
        if elapsed >= next_log_at:
            logger.info(
                f"Waiting for vLLM readiness... ({int(elapsed)}s / {timeout_s}s)"
            )
            next_log_at += 30.0

        time.sleep(2)

    logs = container.logs(tail=30).decode("utf-8", errors="replace")
    raise TimeoutError(
        f"Timed out waiting for vLLM readiness at {last_checked_url} after {timeout_s}s. "
        f"Recent logs:\n{logs}"
    )


def _build_vllm_smoke_test(client: OpenAI, config: VllmConfig):
    def run_smoke_test() -> str:
        test_response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Who are you?"}],
            max_tokens=50,
            temperature=0.5,
        )
        content = test_response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("vLLM smoke test returned a non-text response")
        return content

    return run_smoke_test


def _build_vllm_session(
    client: OpenAI,
    config: VllmConfig,
    base_url: str,
    models_url: str,
    process=None,
) -> BackendSession:
    genner = get_genner("vllm", server_config=config, oai_client=client)
    return BackendSession(
        genner=genner,
        smoke_test=_build_vllm_smoke_test(client, config),
        client=client,
        config=config,
        base_url=base_url,
        models_url=models_url,
        process=process,
    )


@contextmanager
def setup_vllm(
    app_config: AppConfig,
    *,
    endpoint: str = "http://127.0.0.1:8000",
    timeout: int = 500,
) -> Iterator[BackendSession]:
    config = _build_vllm_config(app_config, endpoint=endpoint, timeout=timeout)
    source_model = (
        app_config.model_name.split(":", 1)[1].strip()
        if app_config.model_name.startswith("vllm:")
        else app_config.model_name
    )
    source_model = LEGACY_VLLM_MODEL_ALIASES.get(
        source_model.strip(), source_model.strip()
    )

    endpoint = config.endpoint.rstrip("/")
    base_url = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
    models_url = f"{base_url}/models"
    api_key = config.api_key or "dummy"

    # If a server is already running (e.g. user started one manually), use it.
    if is_http_ready(models_url):
        logger.info(f"Using existing server at {base_url}")
        client = OpenAI(api_key=api_key, base_url=base_url)
        yield _build_vllm_session(client, config, base_url, models_url)
        return

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    scheme = parsed.scheme or "http"
    port = parsed.port or 8000
    network_mode = _get_network_mode()

    host_ip = None if network_mode == "loopback" else _get_host_ip()

    logger.info(
        f"No server detected at {base_url}. "
        f"Starting vLLM Docker container for {source_model} on port {port}..."
    )

    docker_client = DockerClient.from_env()
    container_name = _container_name(port)
    _remove_stale_container(docker_client, container_name)

    vllm_cfg = app_config.vllm
    local_model_path = vllm_cfg.local_model_path if vllm_cfg else None
    lora_adapter_path = vllm_cfg.lora_adapter_path if vllm_cfg else None
    served_model_name = vllm_cfg.served_model_name if vllm_cfg else None

    effective_model, model_host_path, needs_bnb = _resolve_model_for_container(
        source_model,
        local_model_path,
    )

    if model_host_path and not served_model_name and not lora_adapter_path:
        config.model = effective_model

    extra_volumes: list[tuple[str, str]] = []
    if model_host_path:
        extra_volumes.append((model_host_path, LOCAL_MODEL_CONTAINER_PATH))
    if lora_adapter_path:
        extra_volumes.append(
            (
                str(Path(lora_adapter_path).expanduser().resolve()),
                LOCAL_ADAPTER_CONTAINER_PATH,
            )
        )

    vllm_command = _build_container_command(
        effective_model,
        config.gpu_memory_utilization,
        config.chat_template,
        lora_adapter_path=(
            LOCAL_ADAPTER_CONTAINER_PATH if lora_adapter_path is not None else None
        ),
        served_model_name=served_model_name,
        needs_bnb=needs_bnb,
    )
    _start_vllm_container(
        container_name,
        port,
        vllm_command,
        extra_volumes=extra_volumes,
    )

    container = docker_client.containers.get(container_name)
    logger.info(f"Started container '{container_name}' (id={container.short_id})")

    try:
        startup_timeout = max(config.timeout, timeout)

        models_urls = _build_models_urls(
            models_url,
            host_ip,
            port,
            network_mode,
            scheme=scheme,
        )
        if len(models_urls) > 1:
            logger.info(
                f"Will try alternate endpoint {models_urls[1]} if {models_urls[0]} fails"
            )
        elif models_urls[0] != models_url:
            logger.info(
                f"Using host IP endpoint {models_urls[0]} due to "
                f"{VLLM_NETWORK_MODE_ENV}={network_mode}"
            )

        wait_kwargs = {}
        if len(models_urls) == 1:
            wait_kwargs["primary_timeout_s"] = startup_timeout

        working_url = _wait_for_vllm_ready(
            models_urls,
            container,
            startup_timeout,
            **wait_kwargs,
        )
        base_url = working_url.rsplit("/models", 1)[0]
        models_url = working_url
        logger.info(f"vLLM server is ready at {base_url}")

        client = OpenAI(api_key=api_key, base_url=base_url)

        yield _build_vllm_session(
            client,
            config,
            base_url,
            models_url,
            process=container,
        )
    finally:
        logger.info(f"Stopping vLLM container '{container_name}'...")
        try:
            container.stop(timeout=10)
        except Exception as e:
            logger.warning(f"Error stopping container: {e}")
        try:
            container.remove(force=True)
        except Exception as e:
            logger.warning(f"Error removing container: {e}")


__all__ = [
    "LOCAL_ADAPTER_CONTAINER_PATH",
    "LOCAL_MODEL_CONTAINER_PATH",
    "setup_vllm",
    "wait_for_gpu_memory_release",
]
