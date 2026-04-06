import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.observability.collector import MetricsCollector
from src.observability.types import UtilizationSummary


class MetricsCollectorTests(unittest.TestCase):
    def test_snapshot_resources_includes_utilization_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            with (
                patch.object(collector, "_read_gpu_memory_mb", return_value=1024.0),
                patch.object(collector, "_read_host_memory_mb", return_value=2048.0),
                patch.object(collector, "_read_gpu_utilization_pct", return_value=88.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=35.0),
            ):
                snapshot = collector.snapshot_resources()

        self.assertEqual(snapshot.gpu_memory_mb, 1024.0)
        self.assertEqual(snapshot.host_memory_mb, 2048.0)
        self.assertEqual(snapshot.gpu_utilization_pct, 88.0)
        self.assertEqual(snapshot.cpu_utilization_pct, 35.0)

    def test_background_sampling_tracks_peak_and_average(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            # Sequence of readings: GPU 20, 80, 50; CPU 10, 30, 20
            gpu_readings = [20.0, 80.0, 50.0]
            cpu_readings = [10.0, 30.0, 20.0]
            call_count = [0]

            def gpu_read():
                idx = min(call_count[0], len(gpu_readings) - 1)
                return gpu_readings[idx]

            def cpu_read():
                idx = min(call_count[0], len(cpu_readings) - 1)
                call_count[0] += 1
                return cpu_readings[idx]

            with (
                patch.object(collector, "_read_gpu_utilization_pct", side_effect=gpu_read),
                patch.object(collector, "_read_cpu_utilization_pct", side_effect=cpu_read),
            ):
                collector.start_utilization_sampling(interval_seconds=0.05)
                time.sleep(0.25)
                summary = collector.stop_utilization_sampling()

        self.assertIsInstance(summary, UtilizationSummary)
        self.assertIsNotNone(summary.peak_gpu_utilization_pct)
        self.assertIsNotNone(summary.peak_cpu_utilization_pct)
        self.assertIsNotNone(summary.avg_gpu_utilization_pct)
        self.assertIsNotNone(summary.avg_cpu_utilization_pct)
        self.assertGreater(summary.sample_count, 0)
        self.assertEqual(summary.peak_gpu_utilization_pct, 80.0)
        self.assertEqual(summary.peak_cpu_utilization_pct, 30.0)

    def test_stop_utilization_sampling_without_start_returns_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))
            summary = collector.stop_utilization_sampling()

        self.assertIsInstance(summary, UtilizationSummary)
        self.assertIsNone(summary.peak_gpu_utilization_pct)
        self.assertIsNone(summary.peak_cpu_utilization_pct)
        self.assertIsNone(summary.avg_gpu_utilization_pct)
        self.assertIsNone(summary.avg_cpu_utilization_pct)
        self.assertEqual(summary.sample_count, 0)

    def test_disabled_collector_sampling_returns_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(
                run_id="run-123", output_dir=Path(temp_dir), enabled=False
            )
            collector.start_utilization_sampling(interval_seconds=0.05)
            time.sleep(0.1)
            summary = collector.stop_utilization_sampling()

        self.assertIsNone(summary.peak_gpu_utilization_pct)
        self.assertEqual(summary.sample_count, 0)


if __name__ == "__main__":
    unittest.main()
