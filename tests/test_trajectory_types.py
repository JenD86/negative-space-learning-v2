import unittest

from src.typing.trajectory import EpisodeTrajectory, GenerationData


class TrajectoryTypeTests(unittest.TestCase):
    def make_episode(
        self,
        episode_index: int,
        *,
        row_count: int = 1,
        success: bool = True,
        episode_runtime_success: bool | None = None,
        space_freed_kb: float = 128.0,
        partial: bool = False,
        error_message: str | None = None,
        container_overhead_seconds: float | None = None,
        episode_execution_seconds: float | None = None,
        total_inference_ms: float | None = None,
        inference_call_count: int | None = None,
        average_output_tokens_per_second: float | None = None,
        inference_duty_cycle: float | None = None,
        gpu_utilization_pct: float | None = None,
        cpu_utilization_pct: float | None = None,
    ) -> EpisodeTrajectory:
        prompt_responses = [
            {
                "prompt": f"prompt-{episode_index}-{idx}",
                "raw_response": f"response-{episode_index}-{idx}",
                "interaction_type": "explorer",
                "timestamp": "2026-04-06T00:00:00",
                "success": True,
                "error_message": None,
            }
            for idx in range(row_count)
        ]
        return EpisodeTrajectory(
            episode_id=f"ep-{episode_index}",
            generation_id=7,
            episode_index=episode_index,
            prompt_responses=prompt_responses,
            trajectory={"mode_history": []},
            space_freed_kb=space_freed_kb,
            episode_runtime_success=(
                episode_runtime_success
                if episode_runtime_success is not None
                else space_freed_kb > 0.0
            ),
            success=success,
            action_count=3,
            container_variation="variation_1_heavy",
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T00:00:03",
            duration_seconds=3.0,
            partial=partial,
            error_message=error_message,
            space_measurements={"container-a": (1024.0, 1152.0)},
            filesystem_groups=[
                {
                    "filesystem_id": "fs-shared",
                    "container_ids": ["container-a", "container-b"],
                }
            ],
            measurement_errors=["container-b"] if error_message else [],
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=episode_execution_seconds,
            total_inference_ms=total_inference_ms,
            inference_call_count=inference_call_count,
            average_output_tokens_per_second=average_output_tokens_per_second,
            inference_duty_cycle=inference_duty_cycle,
            gpu_utilization_pct=gpu_utilization_pct,
            cpu_utilization_pct=cpu_utilization_pct,
        )

    def test_generation_data_filters_using_generation_threshold(self) -> None:
        generation_data = GenerationData(generation_id=7)
        episode = self.make_episode(
            0,
            success=False,
            episode_runtime_success=True,
            space_freed_kb=5.0,
        )

        generation_data.add_episode(episode)

        self.assertEqual(generation_data.total_successful, 0)
        self.assertEqual(generation_data.total_rows_collected, 0)
        self.assertEqual(generation_data.failed_episodes, [episode])

    def test_generation_data_counts_rows_across_episodes(self) -> None:
        generation_data = GenerationData(generation_id=7)
        generation_data.add_episode(self.make_episode(0, row_count=2, success=True))
        generation_data.add_episode(self.make_episode(1, row_count=3, success=True))
        generation_data.add_episode(self.make_episode(2, row_count=4, success=False))

        self.assertEqual(generation_data.total_episodes_run, 3)
        self.assertEqual(generation_data.total_successful, 2)
        self.assertEqual(generation_data.total_rows_collected, 5)
        self.assertEqual(len(generation_data.successful_episodes), 2)
        self.assertEqual(len(generation_data.failed_episodes), 1)

    def test_uniform_credit_assignment(self) -> None:
        generation_data = GenerationData(generation_id=7)
        generation_data.add_episode(self.make_episode(0, row_count=2, success=True))
        generation_data.add_episode(self.make_episode(1, row_count=1, success=False))
        generation_data.add_episode(self.make_episode(2, row_count=3, success=True))

        rows = generation_data.get_sft_training_rows()

        self.assertEqual(len(rows), 5)
        self.assertEqual({row["episode_id"] for row in rows}, {"ep-0", "ep-2"})

    def test_generation_data_success_rate(self) -> None:
        generation_data = GenerationData(generation_id=7)
        generation_data.add_episode(self.make_episode(0, success=True))
        generation_data.add_episode(self.make_episode(1, success=False))
        generation_data.add_episode(self.make_episode(2, success=False))
        generation_data.add_episode(self.make_episode(3, success=True))

        self.assertEqual(generation_data.success_rate, 0.5)

    def test_episode_trajectory_serialization(self) -> None:
        episode = self.make_episode(4, row_count=2, success=True)

        payload = episode.to_dict()

        self.assertEqual(payload["episode_id"], "ep-4")
        self.assertEqual(payload["generation_id"], 7)
        self.assertEqual(payload["episode_index"], 4)
        self.assertEqual(len(payload["prompt_responses"]), 2)
        self.assertEqual(payload["duration_seconds"], 3.0)
        self.assertEqual(payload["container_variation"], "variation_1_heavy")

    def test_episode_trajectory_retains_dedup_measurement_inputs(self) -> None:
        episode = self.make_episode(5, success=True)

        payload = episode.to_dict()

        self.assertEqual(
            payload["space_measurements"], {"container-a": (1024.0, 1152.0)}
        )
        self.assertEqual(
            payload["filesystem_groups"],
            [
                {
                    "filesystem_id": "fs-shared",
                    "container_ids": ["container-a", "container-b"],
                }
            ],
        )

    def test_episode_partial_and_runtime_success_flags(self) -> None:
        episode = self.make_episode(
            6,
            success=False,
            episode_runtime_success=False,
            partial=True,
            error_message="measurement failed",
        )

        payload = episode.to_dict()

        self.assertFalse(payload["episode_runtime_success"])
        self.assertTrue(payload["partial"])
        self.assertEqual(payload["error_message"], "measurement failed")
        self.assertEqual(payload["measurement_errors"], ["container-b"])

    def test_sft_rows_annotated_with_metadata(self) -> None:
        generation_data = GenerationData(generation_id=7)
        generation_data.add_episode(self.make_episode(0, row_count=2, success=True))

        rows = generation_data.get_sft_training_rows()

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["generation_id"], 7)
            self.assertEqual(row["episode_id"], "ep-0")
            self.assertEqual(row["episode_space_freed_kb"], 128.0)

    def test_episode_trajectory_serialization_includes_observability_fields(
        self,
    ) -> None:
        episode = self.make_episode(
            7,
            success=True,
            container_overhead_seconds=1.25,
            episode_execution_seconds=8.5,
            total_inference_ms=6200.0,
            inference_call_count=4,
            average_output_tokens_per_second=123.4,
            inference_duty_cycle=0.64,
            gpu_utilization_pct=87.0,
            cpu_utilization_pct=42.0,
        )

        payload = episode.to_dict()
        restored = EpisodeTrajectory.from_dict(payload)

        self.assertEqual(payload["container_overhead_seconds"], 1.25)
        self.assertEqual(payload["episode_execution_seconds"], 8.5)
        self.assertEqual(payload["total_inference_ms"], 6200.0)
        self.assertEqual(payload["inference_call_count"], 4)
        self.assertEqual(payload["average_output_tokens_per_second"], 123.4)
        self.assertEqual(payload["inference_duty_cycle"], 0.64)
        self.assertEqual(payload["gpu_utilization_pct"], 87.0)
        self.assertEqual(payload["cpu_utilization_pct"], 42.0)
        self.assertEqual(restored.average_output_tokens_per_second, 123.4)
        self.assertEqual(restored.gpu_utilization_pct, 87.0)

    def test_generation_metadata_includes_throughput_and_utilization_summary(
        self,
    ) -> None:
        generation_data = GenerationData(
            generation_id=7,
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T01:00:00",
        )
        generation_data.add_episode(
            self.make_episode(
                0,
                row_count=2,
                success=True,
                average_output_tokens_per_second=100.0,
                inference_duty_cycle=0.5,
                gpu_utilization_pct=70.0,
                cpu_utilization_pct=30.0,
                container_overhead_seconds=2.0,
                episode_execution_seconds=8.0,
                total_inference_ms=5000.0,
            )
        )
        generation_data.add_episode(
            self.make_episode(
                1,
                row_count=4,
                success=True,
                average_output_tokens_per_second=200.0,
                inference_duty_cycle=0.75,
                gpu_utilization_pct=90.0,
                cpu_utilization_pct=45.0,
                container_overhead_seconds=1.0,
                episode_execution_seconds=9.0,
                total_inference_ms=7500.0,
            )
        )

        metadata = generation_data.to_metadata_dict(run_id="run-123")

        self.assertEqual(metadata["episodes_per_hour"], 2.0)
        self.assertEqual(metadata["successful_rows_per_hour"], 6.0)
        self.assertEqual(metadata["average_output_tokens_per_second"], 150.0)
        self.assertEqual(metadata["average_inference_duty_cycle"], 0.625)
        self.assertEqual(metadata["average_gpu_utilization_pct"], 80.0)
        self.assertEqual(metadata["average_cpu_utilization_pct"], 37.5)
        self.assertEqual(metadata["total_container_overhead_seconds"], 3.0)
        self.assertEqual(metadata["total_inference_seconds"], 12.5)


if __name__ == "__main__":
    unittest.main()
