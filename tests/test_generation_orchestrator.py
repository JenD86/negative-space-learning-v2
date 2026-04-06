import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from scripts.generate_training_data import (
    run_generation,
    run_single_episode,
    save_generation_data,
    select_variation_index,
)
from src.typing.config import AppConfig
from src.typing.trajectory import EpisodeTrajectory, GenerationData


class GenerationOrchestratorTests(unittest.TestCase):
    def make_config(self, base_dir: Path, **generation_overrides: object) -> AppConfig:
        generation_config = {
            "target_successful_rows": 10,
            "max_episodes": 5,
            "container_restart_interval": 50,
            "container_rebuild_interval": 50,
            "success_threshold_kb": 0.0,
            "variation_strategy": "round_robin",
            "variation_random_seed": None,
            "post_rebuild_wait_seconds": 1,
            "population_verification_tolerance": 0.2,
            "checkpoint_every_episode": False,
            "resume_from_checkpoint": False,
            "show_progress": False,
            "resource_snapshot_interval_episodes": 1,
            "generation_output_dir": str(base_dir / "generations"),
            "reset_scratchpad_between_episodes": True,
        }
        generation_config.update(generation_overrides)
        return AppConfig(
            dev=False,
            model_name="claude",
            code_host_cache_path=str(base_dir / "code-host-cache"),
            container_ids=["container-a", "container-b"],
            main_container_idx=0,
            dynamic_container=False,
            docker_compose_dir=str(base_dir / "compose"),
            train_data_save_folder=str(base_dir / "train-data"),
            episode={
                "action_budget": 2,
                "scratchpad_max_chars": 1000,
                "scratchpad_storage_path": str(base_dir / "scratchpad.json"),
                "success_threshold_kb": 0.0,
            },
            generation=generation_config,
            observability={
                "enabled": True,
                "record_inference": False,
                "record_phases": False,
                "record_resources": True,
            },
        )

    def make_episode(
        self,
        episode_index: int,
        *,
        row_count: int = 1,
        success: bool = True,
        episode_runtime_success: bool | None = None,
        partial: bool = False,
        error_message: str | None = None,
        space_freed_kb: float = 64.0,
        container_overhead_seconds: float | None = None,
        episode_execution_seconds: float | None = None,
        total_inference_ms: float | None = None,
        inference_call_count: int | None = None,
        average_output_tokens_per_second: float | None = None,
        inference_duty_cycle: float | None = None,
        gpu_utilization_pct: float | None = None,
        cpu_utilization_pct: float | None = None,
    ) -> EpisodeTrajectory:
        return EpisodeTrajectory(
            episode_id=f"ep-{episode_index}",
            generation_id=0,
            episode_index=episode_index,
            prompt_responses=[
                {
                    "prompt": f"prompt-{episode_index}-{idx}",
                    "raw_response": f"response-{episode_index}-{idx}",
                    "interaction_type": "orchestrator",
                    "timestamp": "2026-04-06T00:00:00",
                    "success": True,
                    "error_message": None,
                }
                for idx in range(row_count)
            ],
            trajectory={"mode_history": []},
            space_freed_kb=space_freed_kb,
            episode_runtime_success=(
                episode_runtime_success
                if episode_runtime_success is not None
                else space_freed_kb > 0.0
            ),
            success=success,
            action_count=2,
            container_variation="variation_1_heavy",
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T00:00:01",
            duration_seconds=1.0,
            partial=partial,
            error_message=error_message,
            space_measurements={"container-a": (100.0, 164.0)},
            filesystem_groups=[
                {
                    "filesystem_id": "fs-shared",
                    "container_ids": ["container-a", "container-b"],
                }
            ],
            measurement_errors=[],
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=episode_execution_seconds,
            total_inference_ms=total_inference_ms,
            inference_call_count=inference_call_count,
            average_output_tokens_per_second=average_output_tokens_per_second,
            inference_duty_cycle=inference_duty_cycle,
            gpu_utilization_pct=gpu_utilization_pct,
            cpu_utilization_pct=cpu_utilization_pct,
        )

    def make_episode_result(
        self,
        *,
        episode_id: str = "episode-1",
        row_count: int = 1,
        success: bool = True,
        episode_runtime_success: bool = True,
        partial: bool = False,
        error_message: str | None = None,
        space_freed_kb: float = 64.0,
    ) -> dict[str, object]:
        return {
            "trajectory": {"mode_history": []},
            "prompt_responses": [
                {
                    "prompt": f"prompt-{idx}",
                    "raw_response": f"response-{idx}",
                    "interaction_type": "explorer",
                    "timestamp": "2026-04-06T00:00:00",
                    "success": True,
                    "error_message": None,
                }
                for idx in range(row_count)
            ],
            "space_freed_kb": space_freed_kb,
            "success": success,
            "episode_runtime_success": episode_runtime_success,
            "episode_id": episode_id,
            "action_count": 2,
            "partial": partial,
            "error_message": error_message,
            "space_measurements": {"container-a": (100.0, 164.0)},
            "filesystem_groups": [
                {
                    "filesystem_id": "fs-shared",
                    "container_ids": ["container-a", "container-b"],
                }
            ],
            "measurement_errors": [],
        }

    def configure_manager_mock(self, manager: MagicMock) -> None:
        manager.get_mixed_cleanup_variations.return_value = [{}, {}, {}, {}, {}]

    def test_run_generation_stops_at_target_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=3,
                max_episodes=10,
            )
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(0, row_count=2, success=True),
                    self.make_episode(1, row_count=2, success=True),
                    self.make_episode(2, row_count=2, success=True),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                )

        self.assertEqual(run_single_episode_mock.call_count, 2)
        self.assertEqual(generation_data.total_rows_collected, 4)
        self.assertEqual(generation_data.total_successful, 2)

    def test_run_generation_stops_at_max_episodes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=100,
                max_episodes=2,
            )
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        0, row_count=0, success=False, space_freed_kb=0.0
                    ),
                    self.make_episode(
                        1, row_count=0, success=False, space_freed_kb=0.0
                    ),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                )

        self.assertEqual(run_single_episode_mock.call_count, 2)
        self.assertEqual(generation_data.total_episodes_run, 2)
        self.assertEqual(generation_data.total_rows_collected, 0)

    def test_variation_round_robin_strategy(self) -> None:
        indices = [select_variation_index("round_robin", idx, 5) for idx in range(8)]

        self.assertEqual(indices, [0, 1, 2, 3, 4, 0, 1, 2])

    def test_variation_random_strategy_with_seed(self) -> None:
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        picks_a = [
            select_variation_index("random", idx, 5, rng=rng_a) for idx in range(6)
        ]
        picks_b = [
            select_variation_index("random", idx, 5, rng=rng_b) for idx in range(6)
        ]

        self.assertEqual(picks_a, picks_b)
        self.assertTrue(all(0 <= pick < 5 for pick in picks_a))

    def test_container_rebuild_at_interval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=100,
                max_episodes=5,
                container_restart_interval=99,
                container_rebuild_interval=2,
            )
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        idx, row_count=0, success=False, space_freed_kb=0.0
                    )
                    for idx in range(5)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                )

        self.assertEqual(manager.rebuild.call_count, 2)

    def test_container_restart_at_interval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=100,
                max_episodes=5,
                container_restart_interval=2,
                container_rebuild_interval=99,
            )
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        idx, row_count=0, success=False, space_freed_kb=0.0
                    )
                    for idx in range(5)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                )

        self.assertEqual(manager.restart.call_count, 2)

    def test_rebuild_interval_takes_precedence_over_restart_overlap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=100,
                max_episodes=6,
                container_restart_interval=2,
                container_rebuild_interval=4,
            )
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        idx, row_count=0, success=False, space_freed_kb=0.0
                    )
                    for idx in range(6)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                )

        self.assertEqual(manager.rebuild.call_count, 1)
        self.assertEqual(manager.restart.call_count, 1)

    @patch("scripts.generate_training_data.run_episode_v2")
    def test_run_single_episode_passes_design_format_episode_id(
        self,
        run_episode_v2: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_1_heavy", expected_kb=15500)
            ]
            manager.verify_population.return_value = {"success": True}

            def _run_episode_side_effect(
                *args: object, **kwargs: object
            ) -> dict[str, object]:
                return self.make_episode_result(episode_id=str(kwargs["episode_id"]))

            run_episode_v2.side_effect = _run_episode_side_effect

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=3,
                episode_index=42,
                variation_index=0,
                run_id="run-123",
            )

        actual_episode_id = str(run_episode_v2.call_args.kwargs.get("episode_id", ""))
        self.assertRegex(actual_episode_id, r"^ep_gen3_0042_\d{10,}$")
        self.assertEqual(trajectory.episode_id, actual_episode_id)

    @patch("scripts.generate_training_data.run_episode_v2")
    def test_container_populate_every_episode(self, run_episode_v2: MagicMock) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_4_large_sparse", expected_kb=19400)
            ]
            manager.verify_population.return_value = {"success": True}
            run_episode_v2.return_value = self.make_episode_result(
                episode_id="ep-0",
                row_count=2,
                success=True,
                episode_runtime_success=True,
                space_freed_kb=64.0,
            )

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=3,
                run_id="run-123",
            )

        manager.populate.assert_called_once_with(3)
        manager.verify_population.assert_called_once_with(19400, 0.2)
        self.assertEqual(trajectory.container_variation, "variation_4_large_sparse")
        self.assertEqual(len(trajectory.prompt_responses), 2)

    @patch("scripts.generate_training_data.run_episode_v2")
    def test_scratchpad_reset_when_configured(self, run_episode_v2: MagicMock) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            scratchpad_path = base_dir / "scratchpad.json"
            scratchpad_path.write_text("stale scratchpad", encoding="utf-8")
            config = self.make_config(base_dir, reset_scratchpad_between_episodes=True)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_1_heavy", expected_kb=15500)
            ]
            manager.verify_population.return_value = {"success": True}
            run_episode_v2.return_value = self.make_episode_result(episode_id="ep-0")

            run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
            )

            self.assertFalse(scratchpad_path.exists())

    @patch("scripts.generate_training_data.run_episode_v2")
    def test_scratchpad_preserved_when_not_configured(
        self, run_episode_v2: MagicMock
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            scratchpad_path = base_dir / "scratchpad.json"
            scratchpad_path.write_text("keep me", encoding="utf-8")
            config = self.make_config(base_dir, reset_scratchpad_between_episodes=False)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_1_heavy", expected_kb=15500)
            ]
            manager.verify_population.return_value = {"success": True}
            run_episode_v2.return_value = self.make_episode_result(episode_id="ep-0")

            run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
            )

            self.assertTrue(scratchpad_path.exists())

    @patch("scripts.generate_training_data.run_episode_v2")
    def test_episode_scratchpad_snapshot_survives_next_reset(
        self, run_episode_v2: MagicMock
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            scratchpad_path = base_dir / "scratchpad.json"
            config = self.make_config(base_dir, reset_scratchpad_between_episodes=True)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_1_heavy", expected_kb=15500)
            ]
            manager.verify_population.return_value = {"success": True}

            def write_scratchpad(*args: object, **kwargs: object) -> dict[str, object]:
                episode_id = str(kwargs["episode_id"])
                scratchpad_path.write_text(
                    json.dumps(
                        {
                            "content": f"notes for {episode_id}",
                            "char_count": len(f"notes for {episode_id}"),
                            "max_chars": 1000,
                            "last_updated": "2026-04-06T00:00:00",
                        }
                    ),
                    encoding="utf-8",
                )
                return self.make_episode_result(episode_id=episode_id)

            run_episode_v2.side_effect = write_scratchpad

            first_episode = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
            )

            snapshot_path = Path(first_episode.trajectory["scratchpad_snapshot_path"])
            snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

            self.assertTrue(snapshot_path.exists())
            self.assertEqual(snapshot_payload["episode_index"], 0)
            self.assertEqual(snapshot_payload["episode_id"], first_episode.episode_id)
            self.assertEqual(
                snapshot_payload["scratchpad"]["content"],
                f"notes for {first_episode.episode_id}",
            )
            self.assertTrue(scratchpad_path.exists())

            run_episode_v2.side_effect = None
            run_episode_v2.return_value = self.make_episode_result(episode_id="ep-1")

            run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=1,
                variation_index=0,
                run_id="run-123",
            )

            self.assertFalse(scratchpad_path.exists())
            self.assertEqual(snapshot_payload["snapshot_reason"], "episode_complete")

    @patch("scripts.generate_training_data.run_episode_v2")
    def test_partial_episode_from_mode_exception_is_recorded(
        self,
        run_episode_v2: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_1_heavy", expected_kb=15500)
            ]
            manager.verify_population.return_value = {"success": True}
            run_episode_v2.return_value = self.make_episode_result(
                episode_id="ep-0",
                success=False,
                episode_runtime_success=False,
                partial=True,
                error_message="mode error",
                space_freed_kb=0.0,
            )

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
            )

        self.assertTrue(trajectory.partial)
        self.assertFalse(trajectory.episode_runtime_success)
        self.assertEqual(trajectory.error_message, "mode error")

    @patch("scripts.generate_training_data.run_episode_v2")
    def test_run_single_episode_converts_episode_exception_to_failed_trajectory(
        self,
        run_episode_v2: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_1_heavy", expected_kb=15500)
            ]
            manager.verify_population.return_value = {"success": True}
            run_episode_v2.side_effect = RuntimeError("df -k / failed")

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
            )

        self.assertFalse(trajectory.success)
        self.assertFalse(trajectory.episode_runtime_success)
        self.assertIn("df -k / failed", trajectory.error_message)
        self.assertEqual(trajectory.prompt_responses, [])

    def test_checkpoint_written_after_each_episode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=100,
                max_episodes=2,
                checkpoint_every_episode=True,
            )
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
                patch(
                    "scripts.generate_training_data.append_episode_jsonl"
                ) as append_episode_jsonl,
                patch(
                    "scripts.generate_training_data.save_generation_checkpoint"
                ) as save_generation_checkpoint,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        0, row_count=0, success=False, space_freed_kb=0.0
                    ),
                    self.make_episode(
                        1, row_count=0, success=False, space_freed_kb=0.0
                    ),
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                )

        self.assertEqual(append_episode_jsonl.call_count, 2)
        self.assertEqual(save_generation_checkpoint.call_count, 2)

    def test_resume_from_checkpoint_restores_progress(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                max_episodes=3,
                resume_from_checkpoint=True,
            )
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.load_generation_checkpoint",
                    return_value={"next_episode_index": 2},
                ),
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.return_value = self.make_episode(
                    2,
                    row_count=0,
                    success=False,
                    space_freed_kb=0.0,
                )

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                )

        first_call = run_single_episode_mock.call_args_list[0]
        self.assertEqual(first_call.kwargs["episode_index"], 2)

    def test_metrics_resource_snapshots_recorded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=100,
                max_episodes=5,
                resource_snapshot_interval_episodes=2,
            )
            metrics_collector = MagicMock()
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        idx, row_count=0, success=False, space_freed_kb=0.0
                    )
                    for idx in range(5)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    metrics_collector=metrics_collector,
                )

        self.assertEqual(metrics_collector.snapshot_resources.call_count, 3)

    @patch("scripts.generate_training_data.run_episode_v2")
    @patch("scripts.generate_training_data.time.perf_counter")
    def test_run_single_episode_records_observability_metrics(
        self,
        perf_counter_mock: MagicMock,
        run_episode_v2: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate.return_value = [
                MagicMock(variation_name="variation_1_heavy", expected_kb=15500)
            ]
            manager.verify_population.return_value = {"success": True}
            manager.get_containers.return_value = [MagicMock(), MagicMock()]
            metrics_collector = MagicMock()
            metrics_collector.summary.side_effect = [
                {
                    "inference_calls": 2,
                    "total_latency_ms": 1000.0,
                    "total_output_tokens": 50,
                },
                {
                    "inference_calls": 5,
                    "total_latency_ms": 4000.0,
                    "total_output_tokens": 350,
                },
            ]
            run_episode_v2.return_value = self.make_episode_result(episode_id="ep-0")
            perf_counter_mock.side_effect = [10.0, 12.0, 12.0, 20.0]

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
                metrics_collector=metrics_collector,
            )

        self.assertEqual(trajectory.container_overhead_seconds, 2.0)
        self.assertEqual(trajectory.episode_execution_seconds, 8.0)
        self.assertEqual(trajectory.total_inference_ms, 3000.0)
        self.assertEqual(trajectory.inference_call_count, 3)
        self.assertEqual(trajectory.average_output_tokens_per_second, 100.0)
        self.assertAlmostEqual(trajectory.inference_duty_cycle, 0.3)

    def test_run_generation_updates_progress_postfix_with_observability_metrics(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_successful_rows=100,
                max_episodes=1,
                resource_snapshot_interval_episodes=1,
            )
            metrics_collector = MagicMock()
            metrics_collector.snapshot_resources.return_value = MagicMock(
                gpu_utilization_pct=82.0,
                cpu_utilization_pct=37.0,
            )
            episode_progress = MagicMock()
            rows_progress = MagicMock()
            with (
                patch(
                    "scripts.generate_training_data.ContainerManager"
                ) as ContainerManager,
                patch(
                    "scripts.generate_training_data.run_single_episode"
                ) as run_single_episode_mock,
                patch(
                    "scripts.generate_training_data._make_progress_bar",
                    side_effect=[episode_progress, rows_progress],
                ),
                patch(
                    "scripts.generate_training_data.time.perf_counter",
                    side_effect=[0.0, 1800.0],
                ),
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.return_value = self.make_episode(
                    0,
                    row_count=20,
                    success=True,
                    average_output_tokens_per_second=240.0,
                    inference_duty_cycle=0.75,
                )

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    metrics_collector=metrics_collector,
                )

        postfix = episode_progress.set_postfix.call_args.args[0]
        row_postfix = rows_progress.set_postfix.call_args.args[0]
        self.assertEqual(postfix["tok/s"], "240.0")
        self.assertEqual(postfix["duty"], "75%")
        self.assertEqual(postfix["gpu"], "82%")
        self.assertEqual(postfix["cpu"], "37%")
        self.assertEqual(postfix["ep/hr"], "2.0")
        self.assertEqual(row_postfix["rows/hr"], "40.0")

    def test_save_generation_data_output_structure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            generation_data = GenerationData(generation_id=4)
            generation_data.add_episode(
                self.make_episode(0, row_count=2, success=True, space_freed_kb=64.0)
            )
            generation_data.add_episode(
                self.make_episode(1, row_count=1, success=False, space_freed_kb=0.0)
            )

            generation_dir = save_generation_data(
                generation_data=generation_data,
                output_dir=base_dir / "generations",
                run_id="run-123",
            )

            metadata_path = generation_dir / "metadata.json"
            sft_rows_path = generation_dir / "sft_training_rows.jsonl"
            all_episodes_path = generation_dir / "all_episodes.jsonl"
            checkpoint_path = generation_dir / "checkpoint.json"
            successful_episode_path = generation_dir / "successful" / "ep-0.json"
            failed_episode_path = generation_dir / "failed" / "ep-1.json"

            self.assertTrue(metadata_path.exists())
            self.assertTrue(sft_rows_path.exists())
            self.assertTrue(all_episodes_path.exists())
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(successful_episode_path.exists())
            self.assertTrue(failed_episode_path.exists())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["generation_id"], 4)
            self.assertEqual(metadata["total_rows_collected"], 2)

            sft_rows = [
                json.loads(line)
                for line in sft_rows_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(sft_rows), 2)
            self.assertEqual({row["episode_id"] for row in sft_rows}, {"ep-0"})


if __name__ == "__main__":
    unittest.main()
