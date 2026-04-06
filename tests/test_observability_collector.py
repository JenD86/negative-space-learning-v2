import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.observability.collector import MetricsCollector


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


if __name__ == "__main__":
    unittest.main()
