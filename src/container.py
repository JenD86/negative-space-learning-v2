import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from docker import DockerClient
from docker.models.containers import Container as DockerContainer
from loguru import logger

from src.tool.docker import get_container_free_disk_space_kb_v2, wait_and_get_container


@dataclass
class ContainerPopulationResult:
    container_id: str
    variation_name: str
    description: str
    expected_kb: int
    success: bool
    error_message: Optional[str] = None


class ContainerManager:
    def __init__(
        self,
        docker_client: DockerClient,
        container_ids: List[str],
        docker_compose_dir: Optional[str] = None,
        post_rebuild_wait_seconds: int = 10,
    ) -> None:
        self.docker_client = docker_client
        self.container_ids = list(container_ids)
        self.docker_compose_dir = docker_compose_dir
        self.post_rebuild_wait_seconds = post_rebuild_wait_seconds

    def get_containers(self) -> List[DockerContainer]:
        return [
            wait_and_get_container(self.docker_client, container_id)
            for container_id in self.container_ids
        ]

    def refresh_container_ids(self) -> List[str]:
        containers = self.docker_client.containers.list()
        refreshed_names: List[str] = []
        for container in containers:
            container_name = container.name or ""
            if (
                "special-learn-compose" in container_name
                and "service" in container_name
            ):
                refreshed_names.append(container_name)
        refreshed = sorted(refreshed_names)
        if len(refreshed) >= 2:
            if "service-b" in refreshed[0] and "service-a" in refreshed[1]:
                refreshed = [refreshed[1], refreshed[0], *refreshed[2:]]
            self.container_ids = refreshed[: len(self.container_ids)]
        return list(self.container_ids)

    def populate(self, variation_index: int) -> List[ContainerPopulationResult]:
        variations = self.get_mixed_cleanup_variations()
        variation = variations[variation_index % len(variations)]
        cleanup_command = (
            "sh -c 'rm -rf /tmp/* /var/cache/* /var/log/* /home/alice/trash "
            "/home/alice/.cache /home/alice/.local/share/Trash /home/alice/old "
            "/home/alice/temp 2>/dev/null || true'"
        )
        results: List[ContainerPopulationResult] = []

        for container_id in self.container_ids:
            try:
                container = self.docker_client.containers.get(container_id)
                container.exec_run(cleanup_command, stdout=False, stderr=False)
                for command in variation["commands"]:
                    container.exec_run(
                        ["sh", "-c", command], stdout=False, stderr=False
                    )
                effective_container_id = container.id or container_id
                results.append(
                    ContainerPopulationResult(
                        container_id=effective_container_id,
                        variation_name=variation["name"],
                        description=variation["description"],
                        expected_kb=variation["expected_kb"],
                        success=True,
                    )
                )
            except Exception as exc:
                logger.error(f"Failed to populate {container_id}: {exc}")
                results.append(
                    ContainerPopulationResult(
                        container_id=container_id,
                        variation_name=variation["name"],
                        description=variation["description"],
                        expected_kb=variation["expected_kb"],
                        success=False,
                        error_message=str(exc),
                    )
                )
        return results

    def _measure_population_kb(self) -> Dict[str, float]:
        command = """
        total=0
        for dir in /tmp /var/log /var/cache /var/tmp /home/alice; do
            if [ -d \"$dir\" ]; then
                size=$(du -sk \"$dir\" 2>/dev/null | awk '{print $1}')
                if [ -n \"$size\" ]; then
                    total=$((total + size))
                fi
            fi
        done
        echo $total
        """
        measurements: Dict[str, float] = {}
        for container_id in self.container_ids:
            container = self.docker_client.containers.get(container_id)
            exec_result = container.exec_run(["sh", "-c", command])
            exit_code, output = self._coerce_exec_result(exec_result)
            if exit_code != 0:
                measurements[container_id] = 0.0
                continue
            output_text = output.decode("utf-8", errors="replace").strip()
            measurements[container_id] = float(output_text) if output_text else 0.0
        return measurements

    def verify_population(
        self,
        expected_kb: int,
        tolerance: float,
    ) -> Dict[str, Any]:
        lower_bound_kb = expected_kb * (1.0 - tolerance)
        upper_bound_kb = expected_kb * (1.0 + tolerance)
        containers: Dict[str, Dict[str, Any]] = {}
        success = True

        for container_id, measured_kb in self._measure_population_kb().items():
            within_tolerance = lower_bound_kb <= measured_kb <= upper_bound_kb
            containers[container_id] = {
                "measured_kb": measured_kb,
                "within_tolerance": within_tolerance,
            }
            success = success and within_tolerance

        return {
            "success": success,
            "expected_kb": expected_kb,
            "tolerance": tolerance,
            "lower_bound_kb": lower_bound_kb,
            "upper_bound_kb": upper_bound_kb,
            "containers": containers,
        }

    def rebuild(self) -> None:
        self._require_compose_dir()
        self._run_compose(["docker", "compose", "down"])
        self._run_compose(["docker", "compose", "up", "-d", "--build"])
        self.refresh_container_ids()
        self._install_procps()
        if not self.verify_ready():
            raise RuntimeError("Containers failed readiness check after rebuild")

    def restart(self) -> None:
        self._require_compose_dir()
        self._run_compose(["docker", "compose", "down"])
        self._run_compose(["docker", "compose", "up", "-d"])
        self.refresh_container_ids()
        self._install_procps()
        if not self.verify_ready():
            raise RuntimeError("Containers failed readiness check after restart")

    def verify_ready(self) -> bool:
        for container_id in self.container_ids:
            wait_and_get_container(self.docker_client, container_id)
        return True

    def measure_free_space(self) -> Dict[str, float]:
        measurements: Dict[str, float] = {}
        for container_id in self.container_ids:
            container = self.docker_client.containers.get(container_id)
            effective_container_id = container.id or container_id
            measurements[effective_container_id] = get_container_free_disk_space_kb_v2(
                container
            )
        return measurements

    def _install_procps(self) -> None:
        for container_id in self.container_ids:
            container = self.docker_client.containers.get(container_id)
            container.exec_run(
                ["sh", "-c", "apk add --no-cache procps"],
                stdout=False,
                stderr=False,
            )

    def _run_compose(self, command: List[str]) -> None:
        subprocess.run(
            command,
            cwd=self.docker_compose_dir,
            capture_output=True,
            text=True,
            check=True,
        )

    def _require_compose_dir(self) -> None:
        if not self.docker_compose_dir:
            raise ValueError("docker_compose_dir is required for compose operations")

    @staticmethod
    def _coerce_exec_result(exec_result: Any) -> tuple[int, bytes]:
        if hasattr(exec_result, "exit_code") and hasattr(exec_result, "output"):
            return int(exec_result.exit_code), bytes(exec_result.output)
        if isinstance(exec_result, tuple) and len(exec_result) >= 2:
            return int(exec_result[0]), bytes(exec_result[1])
        raise TypeError(f"Unsupported exec result type: {type(exec_result)!r}")

    @staticmethod
    def get_mixed_cleanup_variations() -> List[Dict[str, Any]]:
        return [
            {
                "name": "variation_1_heavy",
                "commands": [
                    "mkdir -p /tmp/cleanup /tmp/sessions /var/log/app /var/cache/pkg /tmp/db_temp /home/alice/trash /var/tmp/build",
                    "for i in $(seq 1 40); do dd if=/dev/zero of=/tmp/cleanup/temp_$i.tmp bs=100K count=1 2>/dev/null; done",
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/var/log/app/app_$i.log bs=100K count=1 2>/dev/null; done",
                    "for i in $(seq 1 25); do dd if=/dev/zero of=/var/cache/pkg/cache_$i.dat bs=100K count=1 2>/dev/null; done",
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/tmp/db_temp/table_$i.tmp bs=200K count=1 2>/dev/null; done",
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/home/alice/trash/deleted_$i.bak bs=50K count=1 2>/dev/null; done",
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/tmp/build/obj_$i.o bs=200K count=1 2>/dev/null; done",
                ],
                "expected_kb": 15500,
                "description": "Heavy mix - lots of large files across all categories",
            },
            {
                "name": "variation_2_medium",
                "commands": [
                    "mkdir -p /tmp/work /var/log/system /var/cache/app /tmp/downloads /home/alice/.cache /var/tmp/sql",
                    "for i in $(seq 1 50); do dd if=/dev/zero of=/tmp/work/work_$i.tmp bs=50K count=1 2>/dev/null; done",
                    "for i in $(seq 1 40); do dd if=/dev/zero of=/var/log/system/sys_$i.log bs=50K count=1 2>/dev/null; done",
                    "for i in $(seq 1 35); do dd if=/dev/zero of=/var/cache/app/app_$i.cache bs=75K count=1 2>/dev/null; done",
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/tmp/downloads/download_$i.part bs=150K count=1 2>/dev/null; done",
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/home/alice/.cache/thumb_$i.png bs=20K count=1 2>/dev/null; done",
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/var/tmp/sql/query_$i.tmp bs=75K count=1 2>/dev/null; done",
                ],
                "expected_kb": 11475,
                "description": "Medium mix - balanced file sizes and counts",
            },
            {
                "name": "variation_3_many_small",
                "commands": [
                    "mkdir -p /tmp/fragments /var/log/debug /var/cache/thumbnails /tmp/sessions /home/alice/temp /var/tmp/locks",
                    "for i in $(seq 1 200); do dd if=/dev/zero of=/tmp/fragments/frag_$i.tmp bs=10K count=1 2>/dev/null; done",
                    "for i in $(seq 1 100); do dd if=/dev/zero of=/var/log/debug/debug_$i.log bs=20K count=1 2>/dev/null; done",
                    "for i in $(seq 1 150); do dd if=/dev/zero of=/var/cache/thumbnails/thumb_$i.jpg bs=10K count=1 2>/dev/null; done",
                    "for i in $(seq 1 80); do dd if=/dev/zero of=/tmp/sessions/sess_$i.lock bs=5K count=1 2>/dev/null; done",
                    "for i in $(seq 1 60); do dd if=/dev/zero of=/home/alice/temp/temp_$i.dat bs=30K count=1 2>/dev/null; done",
                    "for i in $(seq 1 100); do dd if=/dev/zero of=/var/tmp/locks/lock_$i.pid bs=5K count=1 2>/dev/null; done",
                ],
                "expected_kb": 8200,
                "description": "Many small files - tests handling of numerous small files",
            },
            {
                "name": "variation_4_large_sparse",
                "commands": [
                    "mkdir -p /tmp/backups /var/log/archives /var/cache/packages /tmp/exports /home/alice/old /var/tmp/dumps",
                    "for i in $(seq 1 5); do dd if=/dev/zero of=/tmp/backups/backup_$i.tar bs=1M count=1 2>/dev/null; done",
                    "for i in $(seq 1 8); do dd if=/dev/zero of=/var/log/archives/archive_$i.log.gz bs=500K count=1 2>/dev/null; done",
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/cache/packages/pkg_$i.deb bs=300K count=1 2>/dev/null; done",
                    "for i in $(seq 1 6); do dd if=/dev/zero of=/tmp/exports/export_$i.sql bs=500K count=1 2>/dev/null; done",
                    "for i in $(seq 1 12); do dd if=/dev/zero of=/home/alice/old/old_$i.bak bs=200K count=1 2>/dev/null; done",
                    "for i in $(seq 1 4); do dd if=/dev/zero of=/var/tmp/dumps/dump_$i.sql bs=500K count=1 2>/dev/null; done",
                ],
                "expected_kb": 19400,
                "description": "Large sparse files - few but large files",
            },
            {
                "name": "variation_5_mixed_realistic",
                "commands": [
                    "mkdir -p /tmp/app_temp /var/log/nginx /var/cache/apt /tmp/.build /home/alice/.local/share/Trash /var/tmp/mysql",
                    "for i in $(seq 1 25); do size=$((RANDOM % 200 + 50)); dd if=/dev/zero of=/tmp/app_temp/temp_$i.tmp bs=${size}K count=1 2>/dev/null; done",
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/var/log/nginx/access_$i.log bs=150K count=1 2>/dev/null; done",
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/var/cache/apt/pkg_$i.deb bs=300K count=1 2>/dev/null; done",
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/tmp/.build/obj_$i.o bs=75K count=1 2>/dev/null; done",
                    "for i in $(seq 1 18); do dd if=/dev/zero of=/home/alice/.local/share/Trash/file_$i.old bs=100K count=1 2>/dev/null; done",
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/tmp/mysql/tmp_table_$i.ibd bs=200K count=1 2>/dev/null; done",
                ],
                "expected_kb": 16000,
                "description": "Realistic mixed scenario - mimics real system",
            },
        ]
