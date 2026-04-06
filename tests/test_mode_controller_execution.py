from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from result import Ok

from src.mode_controller import ModeController
from src.typing.config import AppConfig


def make_app_config() -> AppConfig:
    return AppConfig(
        dev=False,
        model_name="qwen-cleanup-merged",
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=["container-a", "container-b"],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        special_egc={"count": 1, "max_retries": 1},
        strategy_list={"max_retries": 1},
        strategy_code={"count": 1, "max_retries": 1},
        episode={"action_budget": 2},
    )


class ModeControllerExecutionTests(unittest.TestCase):
    @patch("src.mode_controller.run_code_in_con")
    @patch("src.mode_controller.write_code_in_con")
    def test_execute_python_code_uses_container_helpers(
        self,
        write_code_in_con: MagicMock,
        run_code_in_con: MagicMock,
    ) -> None:
        write_code_in_con.return_value = ("/tmp/script.py", "print('hello')")
        run_code_in_con.return_value = Ok("hello")

        genner = MagicMock(collector=None)
        mode_controller = ModeController(genner, make_app_config())

        docker_client = MagicMock()
        container = MagicMock()
        container.id = "container-a"

        mode_controller.initialize_docker(
            docker_client=docker_client,
            container=container,
            host_cache_folder=Path("/tmp/code-host-cache/mode_controller"),
        )

        result = mode_controller.execute_python_code("print('hello')", timeout=12)

        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"], "hello")
        write_code_in_con.assert_called_once()
        run_code_in_con.assert_called_once_with(
            container,
            "/tmp/script.py",
            timeout_seconds=12,
        )

    def test_execute_python_code_fails_without_docker_context(self) -> None:
        genner = MagicMock(collector=None)
        mode_controller = ModeController(genner, make_app_config())

        result = mode_controller.execute_python_code("print('hello')")

        self.assertFalse(result["success"])
        self.assertIn("not initialized", result["stderr"].lower())


if __name__ == "__main__":
    unittest.main()
