import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import toml


class TestTrainLoop(unittest.TestCase):
    def make_config(
        self,
        base_dir: Path,
        **orchestration_overrides: object,
    ):
        from src.typing.config import AppConfig

        orchestration_config = {
            "num_generations": 2,
            "training_window_size": 3,
        }
        orchestration_config.update(orchestration_overrides)

        return AppConfig(
            dev=False,
            model_name="vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            code_host_cache_path=str(base_dir / "code-host-cache"),
            container_ids=["container-a", "container-b"],
            main_container_idx=0,
            dynamic_container=False,
            docker_compose_dir=str(base_dir / "compose"),
            train_data_save_folder=str(base_dir / "train-data"),
            vllm={
                "served_model_name": "nsl-test-loop",
            },
            generation={
                "target_successful_rows": 2,
                "max_episodes": 2,
                "container_restart_interval": 10,
                "container_rebuild_interval": 10,
                "show_progress": False,
                "checkpoint_every_episode": False,
                "resume_from_checkpoint": False,
                "generation_output_dir": str(base_dir / "generations"),
            },
            training={
                "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "max_steps": 5,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-4,
                "warmup_steps": 1,
                "max_seq_length": 1024,
                "adapter_output_dir": str(base_dir / "adapters"),
                "report_to": "none",
            },
            orchestration=orchestration_config,
        )

    def _write_training_rows(self, path: Path, generation_id: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "prompt": f"prompt-{generation_id}",
                "raw_response": f"response-{generation_id}",
                "timestamp": "2026-04-07T00:00:00",
                "interaction_type": "orchestrator",
                "success": True,
                "error_message": None,
                "episode_id": f"ep-gen-{generation_id}",
                "generation_id": generation_id,
                "episode_space_freed_kb": 64.0,
            }
        ]
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_orchestration_config_in_app_config(self) -> None:
        from src.helper import unflatten_toml_dict
        from src.typing.config import AppConfig

        config_dict = toml.loads(
            """
dev = false
model_name = "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
code_host_cache_path = "./code"
container_ids = ["container-a"]
train_data_save_folder = "./train-data"

[generation]
generation_output_dir = "./data/generations"

[training]
base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_output_dir = "./models/adapters"

[orchestration]
num_generations = 2
training_window_size = 3
"""
        )

        config = AppConfig(**unflatten_toml_dict(config_dict))

        self.assertIsNotNone(config.orchestration)
        self.assertEqual(config.orchestration.num_generations, 2)
        self.assertEqual(config.orchestration.training_window_size, 3)

    def test_collect_training_window_uses_latest_n_generations(self) -> None:
        from scripts.run_train_loop import _collect_training_window_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            generation_root = Path(temp_dir)
            for generation_id in range(4):
                self._write_training_rows(
                    generation_root
                    / f"generation_{generation_id}"
                    / "sft_training_rows.jsonl",
                    generation_id,
                )

            training_paths = _collect_training_window_paths(
                generation_root,
                end_generation_id=3,
                window_size=3,
            )

        self.assertEqual(
            training_paths,
            [
                generation_root / "generation_1" / "sft_training_rows.jsonl",
                generation_root / "generation_2" / "sft_training_rows.jsonl",
                generation_root / "generation_3" / "sft_training_rows.jsonl",
            ],
        )

    @patch("scripts.run_train_loop.train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_first_generation_without_adapter(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            served_adapters: list[str | None] = []

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
            ) -> Path:
                served_adapters.append(generation_config.vllm.lora_adapter_path)
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.return_value = base_dir / "adapters" / "after_generation_0"

            run_loop(config, run_id="run-123", docker_client=MagicMock())

        self.assertEqual(served_adapters[0], None)

    @patch("scripts.run_train_loop.train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_restarts_backend_with_new_adapter(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            served_adapters: list[str | None] = []
            adapter_dir = base_dir / "adapters" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
            ) -> Path:
                served_adapters.append(generation_config.vllm.lora_adapter_path)
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.return_value = adapter_dir

            run_loop(config, run_id="run-123", docker_client=MagicMock())

        self.assertEqual(served_adapters, [None, str(adapter_dir)])
        self.assertEqual(mock_train_sft.call_args.kwargs["export_format"], "peft")

    @patch("scripts.run_train_loop.train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_uses_merged_export_as_vllm_local_model(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.training.export_format = "merged_16bit"
            served_adapters: list[str | None] = []
            served_local_models: list[str | None] = []
            merged_model_dir = base_dir / "adapters" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
            ) -> Path:
                served_adapters.append(generation_config.vllm.lora_adapter_path)
                served_local_models.append(generation_config.vllm.local_model_path)
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.return_value = merged_model_dir

            run_loop(config, run_id="run-123", docker_client=MagicMock())

        self.assertEqual(served_adapters, [None, None])
        self.assertEqual(served_local_models, [None, str(merged_model_dir)])
        self.assertEqual(
            mock_train_sft.call_args.kwargs["export_format"],
            "merged_16bit",
        )

    @patch("scripts.run_train_loop.train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_two_generations(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            events: list[str] = []
            adapter_dir = base_dir / "adapters" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
            ) -> Path:
                events.append(
                    f"generation:{generation_id}:{generation_config.vllm.lora_adapter_path}"
                )
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir

            def train_side_effect(*args, **kwargs) -> Path:
                training_paths = kwargs["training_data_paths"]
                events.append(
                    f"train:{Path(kwargs['output_dir']).name}:{training_paths}"
                )
                return adapter_dir

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = train_side_effect

            results = run_loop(config, run_id="run-123", docker_client=MagicMock())
            summary_path = (
                Path(config.generation.generation_output_dir)
                / "orchestration_summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(
            events,
            [
                "generation:0:None",
                f"train:after_generation_0:['{base_dir / 'generations' / 'generation_0' / 'sft_training_rows.jsonl'}']",
                f"generation:1:{adapter_dir}",
            ],
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(summary["num_generations"], 2)
        self.assertEqual(summary["training_window_size"], 3)
        self.assertEqual(
            summary["generations"][1]["served_adapter_dir"], str(adapter_dir)
        )

    @patch("scripts.run_train_loop.train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_skips_training_after_final_generation(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
            ) -> Path:
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.return_value = base_dir / "adapters" / "after_generation_0"

            run_loop(config, run_id="run-123", docker_client=MagicMock())

        mock_train_sft.assert_called_once()


if __name__ == "__main__":
    unittest.main()
