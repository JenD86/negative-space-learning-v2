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


def _get_host_ip() -> str:
    """Get a non-loopback IP address for the host.

    On some systems (e.g., NixOS/WSL2), Docker's network=host mode has a broken
    loopback interface, so we need to use the host's actual network IP.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _is_bnb_model(model: str) -> bool:
    lower = model.lower()
    return "bnb-4bit" in lower or "bnb-8bit" in lower


def _normalize_vllm_model(model: str) -> str:
    normalized_model = model.strip()
    legacy_model_aliases = {
        "qwen2.5-coder:7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5:7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    }
    return legacy_model_aliases.get(normalized_model, normalized_model)


def _build_vllm_config(
    app_config: AppConfig, endpoint: str, timeout: int
) -> VllmConfig:
    config = VllmConfig()
    if app_config.model_name.startswith("vllm:"):
        config.model = app_config.model_name.split(":", 1)[1].strip()

    resolved_model = _normalize_vllm_model(config.model)
    if resolved_model != config.model.strip():
        logger.info(f"Resolved model alias '{config.model}' to '{resolved_model}'")
    config.model = resolved_model
    config.endpoint = endpoint
    config.timeout = timeout
    config.gpu_memory_utilization = app_config.gpu_memory_utilization
    config.temperature = app_config.inference.temperature
    config.max_tokens = app_config.inference.max_tokens
    if app_config.vllm is not None and app_config.vllm.chat_template_path:
        chat_template_path = Path(app_config.vllm.chat_template_path)
        config.chat_template = chat_template_path.read_text(encoding="utf-8")
    return config


def _build_container_command(
    model: str,
    gpu_memory_utilization: float,
    chat_template: str | None = None,
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
    if _is_bnb_model(model):
        cmd.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
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
) -> None:
    """Start a vLLM container using the docker CLI.

    We use subprocess instead of the Python SDK because the SDK doesn't
    support CDI (Container Device Interface) for GPU passthrough, which
    is required on NixOS/WSL2.
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
        VLLM_IMAGE,
        *vllm_command,
    ]

    logger.info(f"Docker command: {' '.join(docker_cmd)}")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start vLLM container: {result.stderr.strip()}")


def _wait_for_vllm_ready(
    models_urls: str | list[str],
    container,
    timeout_s: int,
    loopback_timeout_s: int = 60,
) -> str:
    """Wait for vLLM to become ready, trying multiple endpoints.

    Returns the first URL that becomes ready.

    Args:
        models_urls: List of URLs to try (e.g., loopback and host IP)
        container: Docker container handle
        timeout_s: Total timeout for all endpoints
        loopback_timeout_s: Timeout for first (loopback) endpoint before trying alternates
    """
    if isinstance(models_urls, str):
        models_urls = [models_urls]

    deadline = time.time() + timeout_s
    start_time = time.time()
    next_log_at = 30.0

    primary_url = models_urls[0]
    alternate_urls = models_urls[1:]

    # First try the primary URL with a shorter timeout
    primary_deadline = time.time() + min(loopback_timeout_s, timeout_s)
    while time.time() < primary_deadline:
        if is_http_ready(primary_url):
            return primary_url

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

    # If primary timed out, switch to alternate and continue polling
    if alternate_urls:
        current_url = alternate_urls[0]
        logger.info(f"Loopback unreachable, switching to alternate: {current_url}")
    else:
        current_url = primary_url

    while time.time() < deadline:
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
        f"Timed out waiting for vLLM readiness at {current_url} after {timeout_s}s. "
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

    endpoint = config.endpoint.rstrip("/")
    base_url = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
    models_url = f"{base_url}/models"
    api_key = config.api_key or "dummy"
    client = OpenAI(api_key=api_key, base_url=base_url)

    # If a server is already running (e.g. user started one manually), use it.
    if is_http_ready(models_url):
        logger.info(f"Using existing server at {base_url}")
        yield _build_vllm_session(client, config, base_url, models_url)
        return

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    port = parsed.port or 8000

    # On some systems (NixOS/WSL2), Docker's network=host mode has a broken
    # loopback interface. We detect this by trying loopback first, then fall
    # back to the host's actual IP if loopback fails.
    host_ip = _get_host_ip()

    logger.info(
        f"No server detected at {base_url}. "
        f"Starting vLLM Docker container for {config.model} on port {port}..."
    )

    docker_client = DockerClient.from_env()
    container_name = _container_name(port)
    _remove_stale_container(docker_client, container_name)

    vllm_command = _build_container_command(
        config.model,
        config.gpu_memory_utilization,
        config.chat_template,
    )
    _start_vllm_container(container_name, port, vllm_command)

    # Get a handle to the container for lifecycle management
    container = docker_client.containers.get(container_name)
    logger.info(f"Started container '{container_name}' (id={container.short_id})")

    try:
        startup_timeout = max(config.timeout, timeout)

        # Build endpoints to try: original endpoint first, then host IP
        models_urls = [models_url]
        if host_ip != "127.0.0.1":
            host_base_url = f"http://{host_ip}:{port}/v1"
            host_models_url = f"{host_base_url}/models"
            models_urls.append(host_models_url)
            logger.info(
                f"Will try alternate endpoint {host_models_url} if loopback fails"
            )

        working_url = _wait_for_vllm_ready(models_urls, container, startup_timeout)
        base_url = working_url.rsplit("/models", 1)[0]
        models_url = working_url
        logger.info(f"vLLM server is ready at {base_url}")

        # Update client with working base_url
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


__all__ = ["setup_vllm"]
