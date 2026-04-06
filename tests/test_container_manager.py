import unittest
from unittest.mock import MagicMock, call, patch

from src.container import ContainerManager


class ExecResult:
    def __init__(self, exit_code: int = 0, output: bytes = b"") -> None:
        self.exit_code = exit_code
        self.output = output

    def __iter__(self):
        yield self.exit_code
        yield self.output


class ContainerManagerTests(unittest.TestCase):
    def make_manager(
        self, exec_return: bytes = b"ok"
    ) -> tuple[ContainerManager, MagicMock, MagicMock, MagicMock]:
        docker_client = MagicMock()
        container_a = MagicMock()
        container_a.id = "container-a"
        container_a.name = "special-learn-compose_service-a_1"
        container_a.status = "running"
        container_a.exec_run.return_value = ExecResult(0, exec_return)

        container_b = MagicMock()
        container_b.id = "container-b"
        container_b.name = "special-learn-compose_service-b_1"
        container_b.status = "running"
        container_b.exec_run.return_value = ExecResult(0, exec_return)

        containers_by_id = {
            "container-a": container_a,
            "container-b": container_b,
            "special-learn-compose_service-a_1": container_a,
            "special-learn-compose_service-b_1": container_b,
            "special-learn-compose_service-a_9": container_a,
            "special-learn-compose_service-b_9": container_b,
        }
        docker_client.containers.get.side_effect = containers_by_id.__getitem__
        docker_client.containers.list.return_value = [container_b, container_a]

        manager = ContainerManager(
            docker_client=docker_client,
            container_ids=["container-a", "container-b"],
            docker_compose_dir="/tmp/docker-compose",
            post_rebuild_wait_seconds=7,
        )
        return manager, docker_client, container_a, container_b

    def test_populate_applies_variation(self) -> None:
        manager, _, container_a, container_b = self.make_manager()

        with patch.object(
            manager,
            "_measure_population_kb",
            return_value={"container-a": 100.0, "container-b": 110.0},
        ):
            results, baseline = manager.populate(0)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].variation_name, "variation_1_heavy")
        self.assertEqual(results[0].expected_kb, 15500)
        # 1 cleanup + 7 variation commands = 8 exec_run calls per container
        self.assertEqual(container_a.exec_run.call_count, 8)
        self.assertEqual(container_b.exec_run.call_count, 8)

    def test_populate_returns_baseline_measurements(self) -> None:
        manager, _, _, _ = self.make_manager()

        with patch.object(
            manager,
            "_measure_population_kb",
            return_value={"container-a": 50000.0, "container-b": 51000.0},
        ):
            results, baseline = manager.populate(0)

        self.assertEqual(baseline, {"container-a": 50000.0, "container-b": 51000.0})

    def test_verify_population_delta_outside_tolerance_fails(self) -> None:
        """Delta (measured - baseline) is too small relative to expected_kb."""
        manager, _, _, _ = self.make_manager()
        baseline_kb = {"container-a": 50000.0, "container-b": 50000.0}
        with patch.object(
            manager,
            "_measure_population_kb",
            return_value={"container-a": 50300.0, "container-b": 50310.0},
        ):
            report = manager.verify_population(
                expected_kb=1000, tolerance=0.2, baseline_kb=baseline_kb
            )

        self.assertFalse(report["success"])
        self.assertFalse(report["containers"]["container-a"]["within_tolerance"])
        self.assertAlmostEqual(report["containers"]["container-a"]["delta_kb"], 300.0)
        self.assertEqual(report["lower_bound_kb"], 800.0)
        self.assertEqual(report["upper_bound_kb"], 1200.0)

    def test_verify_population_delta_within_tolerance_passes(self) -> None:
        """Delta matches expected_kb despite large base content in containers."""
        manager, _, _, _ = self.make_manager()
        # Simulate 50MB base content (venv, documents, etc.)
        baseline_kb = {"container-a": 50000.0, "container-b": 50000.0}
        # After population: base + variation files
        with patch.object(
            manager,
            "_measure_population_kb",
            return_value={"container-a": 65500.0, "container-b": 65000.0},
        ):
            report = manager.verify_population(
                expected_kb=15500, tolerance=0.2, baseline_kb=baseline_kb
            )

        self.assertTrue(report["success"])
        self.assertTrue(report["containers"]["container-a"]["within_tolerance"])
        self.assertTrue(report["containers"]["container-b"]["within_tolerance"])
        self.assertAlmostEqual(
            report["containers"]["container-a"]["delta_kb"], 15500.0
        )
        self.assertAlmostEqual(
            report["containers"]["container-a"]["baseline_kb"], 50000.0
        )

    def test_verify_population_reports_per_container_details(self) -> None:
        """One container passes, one fails — overall result is failure."""
        manager, _, _, _ = self.make_manager()
        baseline_kb = {"container-a": 50000.0, "container-b": 50000.0}
        with patch.object(
            manager,
            "_measure_population_kb",
            return_value={"container-a": 65500.0, "container-b": 52000.0},
        ):
            report = manager.verify_population(
                expected_kb=15500, tolerance=0.2, baseline_kb=baseline_kb
            )

        self.assertFalse(report["success"])
        self.assertTrue(report["containers"]["container-a"]["within_tolerance"])
        self.assertFalse(report["containers"]["container-b"]["within_tolerance"])

    @patch("src.container.subprocess.run")
    def test_rebuild_calls_compose_with_build_flag(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        with (
            patch.object(manager, "refresh_container_ids") as refresh_container_ids,
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.rebuild()

        subprocess_run.assert_has_calls(
            [
                call(
                    ["docker", "compose", "down"],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                ),
                call(
                    ["docker", "compose", "up", "-d", "--build"],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                ),
            ]
        )
        refresh_container_ids.assert_called_once()

    @patch("src.container.subprocess.run")
    def test_restart_calls_compose_without_build(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        with (
            patch.object(manager, "refresh_container_ids") as refresh_container_ids,
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.restart()

        subprocess_run.assert_has_calls(
            [
                call(
                    ["docker", "compose", "down"],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                ),
                call(
                    ["docker", "compose", "up", "-d"],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                ),
            ]
        )
        refresh_container_ids.assert_called_once()

    def test_refresh_container_ids_after_rebuild_or_restart(self) -> None:
        manager, docker_client, _, _ = self.make_manager()
        refreshed_a = MagicMock()
        refreshed_a.name = "special-learn-compose_service-a_9"
        refreshed_b = MagicMock()
        refreshed_b.name = "special-learn-compose_service-b_9"
        docker_client.containers.list.return_value = [
            refreshed_a,
            refreshed_b,
        ]

        refreshed = manager.refresh_container_ids()

        self.assertEqual(
            refreshed,
            [
                "special-learn-compose_service-a_9",
                "special-learn-compose_service-b_9",
            ],
        )
        self.assertEqual(manager.container_ids, refreshed)

    @patch("src.container.subprocess.run")
    def test_rebuild_installs_procps_for_readiness_probe(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, container_a, container_b = self.make_manager()
        with (
            patch.object(
                manager, "refresh_container_ids", return_value=manager.container_ids
            ),
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.rebuild()

        container_a.exec_run.assert_any_call(
            ["sh", "-c", "apk add --no-cache procps"],
            stdout=False,
            stderr=False,
        )
        container_b.exec_run.assert_any_call(
            ["sh", "-c", "apk add --no-cache procps"],
            stdout=False,
            stderr=False,
        )

    @patch("src.container.time.sleep")
    @patch("src.container.subprocess.run")
    def test_rebuild_does_not_use_fixed_sleep_before_verify_ready(
        self,
        subprocess_run: MagicMock,
        sleep: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        with (
            patch.object(manager, "refresh_container_ids"),
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.rebuild()

        sleep.assert_not_called()

    @patch("src.container.get_container_free_disk_space_kb_v2")
    def test_measure_free_space_delegates_to_v2(
        self,
        get_container_free_disk_space_kb_v2: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        get_container_free_disk_space_kb_v2.side_effect = [123.0, 456.0]

        measurements = manager.measure_free_space()

        self.assertEqual(measurements, {"container-a": 123.0, "container-b": 456.0})

    def test_get_mixed_cleanup_variations_has_5(self) -> None:
        manager, _, _, _ = self.make_manager()

        variations = manager.get_mixed_cleanup_variations()

        self.assertEqual(len(variations), 5)

    @patch("src.container.wait_and_get_container")
    def test_verify_ready_checks_all_containers(
        self,
        wait_and_get_container: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()

        ready = manager.verify_ready()

        self.assertTrue(ready)
        wait_and_get_container.assert_has_calls(
            [
                call(manager.docker_client, "container-a"),
                call(manager.docker_client, "container-b"),
            ]
        )


    def test_cleanup_command_includes_var_tmp_and_dotfiles(self) -> None:
        manager, _, container_a, _ = self.make_manager()
        with patch.object(
            manager,
            "_measure_population_kb",
            return_value={"container-a": 0.0, "container-b": 0.0},
        ):
            manager.populate(0)

        # Inspect the first exec_run call (cleanup command)
        cleanup_call = container_a.exec_run.call_args_list[0]
        cleanup_cmd = cleanup_call[0][0]  # positional arg
        self.assertIn("/var/tmp/", cleanup_cmd)
        self.assertIn("/tmp/.[!.]*", cleanup_cmd)


if __name__ == "__main__":
    unittest.main()
