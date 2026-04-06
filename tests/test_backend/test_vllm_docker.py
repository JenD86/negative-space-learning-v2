import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.genner.Base import Genner
from src.typing.config import AppConfig


def make_app_config(
    model_name: str,
    gpu_mem: float = 0.85,
    chat_template_path: str | None = None,
) -> AppConfig:
    return AppConfig(
        dev=False,
        model_name=model_name,
        gpu_memory_utilization=gpu_mem,
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        special_egc=AppConfig.SpecialEGCConfig(count=1, max_retries=1),
        strategy_list=AppConfig.StrategyListConfig(max_retries=1),
        strategy_code=AppConfig.StrategyCodeConfig(count=1, max_retries=1),
        vllm=AppConfig.VllmConfig(chat_template_path=chat_template_path)
        if chat_template_path is not None
        else None,
    )


class TestBnbModelDetection(unittest.TestCase):
    def test_detects_bnb_4bit_model(self) -> None:
        from src.backend.vllm import _is_bnb_model

        self.assertTrue(_is_bnb_model("unsloth/Qwen2.5-Coder-7B-bnb-4bit"))

    def test_detects_bnb_suffix(self) -> None:
        from src.backend.vllm import _is_bnb_model

        self.assertTrue(_is_bnb_model("some-org/model-bnb-8bit"))

    def test_rejects_non_bnb_model(self) -> None:
        from src.backend.vllm import _is_bnb_model

        self.assertFalse(_is_bnb_model("Qwen/Qwen2.5-Coder-7B-Instruct"))

    def test_rejects_gguf_model(self) -> None:
        from src.backend.vllm import _is_bnb_model

        self.assertFalse(_is_bnb_model("bartowski/model-GGUF"))


class TestBuildVllmDockerConfig(unittest.TestCase):
    def test_get_network_mode_defaults_to_auto(self) -> None:
        from src.backend.vllm import _get_network_mode

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_get_network_mode(), "auto")

    def test_get_network_mode_falls_back_on_invalid_value(self) -> None:
        from src.backend.vllm import _get_network_mode

        with patch.dict(os.environ, {"NSL_VLLM_NETWORK_MODE": "invalid"}, clear=True):
            self.assertEqual(_get_network_mode(), "auto")

    def test_build_vllm_config_loads_chat_template_from_path(self) -> None:
        from src.backend.vllm import _build_vllm_config

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write("{{ messages[0]['content'] }}")
            handle.flush()

            config = make_app_config(
                "vllm:unsloth/Qwen2.5-Coder-7B-bnb-4bit",
                chat_template_path=handle.name,
            )

            vllm_config = _build_vllm_config(
                config,
                endpoint="http://127.0.0.1:8000",
                timeout=500,
            )

        self.assertEqual(vllm_config.chat_template, "{{ messages[0]['content'] }}")

    def test_builds_container_command_for_standard_model(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command("Qwen/Qwen2.5-7B-Instruct", 0.85)
        self.assertIn("--model", cmd)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", cmd)
        self.assertIn("--gpu-memory-utilization", cmd)
        self.assertIn("0.85", cmd)
        self.assertNotIn("--quantization", cmd)

    def test_builds_container_command_for_bnb_model(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command("unsloth/Qwen2.5-Coder-7B-bnb-4bit", 0.85)
        self.assertIn("--quantization", cmd)
        self.assertIn("bitsandbytes", cmd)
        self.assertIn("--load-format", cmd)

    def test_builds_container_command_with_chat_template(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
            0.85,
            "{{ messages[0]['content'] }}",
        )

        self.assertIn("--chat-template", cmd)
        self.assertIn("{{ messages[0]['content'] }}", cmd)


class TestVllmContainerLifecycle(unittest.TestCase):
    """Tests that setup_vllm correctly manages the Docker container lifecycle."""

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_reuses_existing_server(
        self,
        mock_http: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """When a server is already running, no container should be created."""
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config) as session:
            self.assertIsNotNone(session.genner)
            self.assertIsNone(session.process)

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_creates_and_removes_container(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """When no server exists, a Docker container should be created and removed on exit."""
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_container.id = "abc123"
        mock_docker = MagicMock()
        # First get() is stale check (no stale), second is post-creation
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config) as session:
            self.assertIsNotNone(session.genner)
            mock_start.assert_called_once()

        wait_urls = mock_wait.call_args.args[0]
        self.assertEqual(
            wait_urls,
            [
                "http://127.0.0.1:8000/v1/models",
                "http://10.0.0.8:8000/v1/models",
            ],
        )
        self.assertEqual(mock_openai_cls.call_count, 1)

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_hostip_mode_uses_only_host_ip_endpoint(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_container.id = "abc123"
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker
        mock_wait.return_value = "http://10.0.0.8:8000/v1/models"

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with patch.dict(os.environ, {"NSL_VLLM_NETWORK_MODE": "hostip"}, clear=True):
            with setup_vllm(config) as session:
                self.assertIsNotNone(session.genner)

        self.assertEqual(
            mock_wait.call_args.args[0], ["http://10.0.0.8:8000/v1/models"]
        )
        self.assertEqual(
            mock_openai_cls.call_args.kwargs["base_url"], "http://10.0.0.8:8000/v1"
        )
        self.assertEqual(
            mock_wait.call_args.kwargs["primary_timeout_s"],
            max(config.inference.timeout, 500),
        )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_removes_stale_container_before_creating(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """If a stale container with the same name exists, it should be removed first."""
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)

        # First get() call is for stale removal, second is for getting the new container
        stale_container = MagicMock()
        stale_container.name = "nsl-vllm-8000"
        new_container = MagicMock()
        new_container.name = "nsl-vllm-8000"
        new_container.id = "new123"

        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [stale_container, new_container]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config) as session:
            stale_container.remove.assert_called_once_with(force=True)
            mock_start.assert_called_once()

        new_container.stop.assert_called_once()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_container_uses_bnb_flags_for_bnb_model(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """BnB quantized models should get --quantization and --load-format flags."""
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:unsloth/Qwen2.5-Coder-7B-bnb-4bit")

        with setup_vllm(config) as session:
            start_call = mock_start.call_args
            vllm_command = start_call[0][2]  # third positional arg
            self.assertIn("--quantization", vllm_command)
            self.assertIn("bitsandbytes", vllm_command)
            self.assertIn("--load-format", vllm_command)

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_container_uses_chat_template_from_config(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write("{{ messages[0]['content'] }}")
            handle.flush()
            config = make_app_config(
                "vllm:unsloth/Qwen2.5-Coder-7B-bnb-4bit",
                chat_template_path=handle.name,
            )

            with setup_vllm(config):
                start_call = mock_start.call_args
                vllm_command = start_call[0][2]
                self.assertIn("--chat-template", vllm_command)
                self.assertIn("{{ messages[0]['content'] }}", vllm_command)

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_cleanup_on_startup_failure(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """If readiness check fails, container should still be cleaned up."""
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_wait.side_effect = TimeoutError("Timed out")
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with self.assertRaises(TimeoutError):
            with setup_vllm(config):
                pass

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()


class TestStartVllmContainer(unittest.TestCase):
    @patch("src.backend.vllm.subprocess.run")
    def test_calls_docker_run_with_gpu_and_volume(self, mock_run: MagicMock) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=0)
        _start_vllm_container("nsl-vllm-8000", 8000, ["--model", "test"])

        call_args = mock_run.call_args[0][0]
        self.assertIn("docker", call_args)
        self.assertIn("nvidia.com/gpu=all", call_args)
        self.assertIn("--ipc=host", call_args)
        self.assertIn("--model", call_args)

    @patch("src.backend.vllm.subprocess.run")
    def test_raises_on_docker_failure(self, mock_run: MagicMock) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=1, stderr="no space left")
        with self.assertRaises(RuntimeError):
            _start_vllm_container("nsl-vllm-8000", 8000, ["--model", "test"])


class TestWaitForVllmReady(unittest.TestCase):
    @patch("src.backend.vllm.is_http_ready")
    @patch("src.backend.vllm.time")
    def test_returns_when_http_ready(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        mock_time.time.side_effect = [0, 0, 0, 1]
        mock_http.return_value = True
        mock_container = MagicMock()
        mock_container.status = "running"

        _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready", return_value=False)
    @patch("src.backend.vllm.time")
    def test_raises_on_timeout(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        # time.time() is called multiple times per loop iteration.
        # Use a counter to return values that eventually exceed the deadline.
        call_count = 0

        def fake_time():
            nonlocal call_count
            call_count += 1
            return 0 if call_count == 1 else (30 if call_count < 5 else 91)

        mock_time.time.side_effect = fake_time
        mock_time.sleep = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.reload = MagicMock()
        mock_container.logs.return_value = b"loading..."

        with self.assertRaises(TimeoutError):
            _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready", return_value=False)
    @patch("src.backend.vllm.time")
    def test_raises_on_container_exit(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        mock_time.time.side_effect = [0, 0, 0, 1]
        mock_time.sleep = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_container.reload = MagicMock()
        mock_container.logs.return_value = b"some error"

        with self.assertRaises(RuntimeError, msg="container exited"):
            _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)


if __name__ == "__main__":
    unittest.main()
