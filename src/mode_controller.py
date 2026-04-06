from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from enum import Enum
from pathlib import Path

from docker import DockerClient
from docker.models.containers import Container as DockerContainer
from loguru import logger
from result import Ok, Err
import re
import time

from src.action_system import ActionBudget, PromptResponsePair
from src.observability.types import PhaseMetric
from src.scratchpad import CrossEpisodeScratchpad
from src.genner.Base import Genner
from src.tool.docker import run_code_in_con, write_code_in_con
from src.typing.config import AppConfig
from src.typing.message import Message
from datetime import datetime


class Mode(Enum):
    ORCHESTRATOR = "orchestrator"
    INVESTIGATOR = "investigator"
    EXPLORER = "explorer"
    RECORDER = "recorder"


@dataclass
class OrchestratorDecision:
    """Parsed orchestrator response with mode delegation."""

    target_mode: Mode
    instruction: str
    reasoning: str
    raw_response: str
    duration_ms: float = 0.0


@dataclass
class ModeResult:
    """Standardized result format across all modes."""

    mode: Mode
    success: bool
    content: str  # Main response content
    structured_data: Dict[str, Any]  # Parsed data (findings, space_freed, etc.)
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    code_executed: Optional[str] = None  # For investigator/explorer modes
    prompt_response_pair: Optional[PromptResponsePair] = None


@dataclass
class EpisodeState:
    """Track current episode state across mode transitions."""

    episode_id: str
    current_step: int
    mode_history: List[Dict[str, Any]]
    total_space_freed: float = 0.0

    def add_mode_execution(
        self,
        mode: Mode,
        instruction: str,
        result: ModeResult,
        orchestrator_duration_ms: float = 0.0,
    ):
        """Record a mode execution in history."""
        self.mode_history.append(
            {
                "step": self.current_step,
                "mode": mode.value,
                "instruction": instruction,
                "success": result.success,
                "content": result.content[:200] + "..."
                if len(result.content) > 200
                else result.content,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": result.duration_ms,
                "orchestrator_duration_ms": orchestrator_duration_ms,
            }
        )
        self.current_step += 1


class ModeController:
    """Unified AI system operating in different modes with single persona."""

    def __init__(self, genner: Genner, config: AppConfig):
        self.genner = genner
        self.config = config
        self.metrics_collector = genner.collector
        self.docker_client: Optional[DockerClient] = None
        self.execution_container: Optional[DockerContainer] = None
        self.code_host_cache_folder: Optional[Path] = None

        # Initialize episode components
        episode_config = config.episode or config.EpisodeConfig()
        self.budget = ActionBudget(total_budget=episode_config.action_budget)

        storage_path = episode_config.scratchpad_storage_path
        storage_path_obj = Path(storage_path) if storage_path else None

        self.scratchpad = CrossEpisodeScratchpad(
            max_chars=episode_config.scratchpad_max_chars,
            storage_path=storage_path_obj,
        )

        # Episode state tracking
        self.episode_state = None
        self.current_mode = Mode.ORCHESTRATOR

    def initialize_docker(
        self,
        docker_client: DockerClient,
        container: DockerContainer,
        host_cache_folder: Path,
    ) -> None:
        self.docker_client = docker_client
        self.execution_container = container
        self.code_host_cache_folder = host_cache_folder

    def _message(self, role: str, content: str, phase: str) -> Message:
        meta: Dict[str, Any] = {"phase": phase}
        if self.episode_state is not None:
            meta["episode_id"] = self.episode_state.episode_id
        return {"role": role, "content": content, "meta": meta}

    def _record_phase_metric(
        self,
        phase_name: str,
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        if self.metrics_collector is None:
            return
        episode_id = None
        if self.episode_state is not None:
            episode_id = self.episode_state.episode_id
        self.metrics_collector.record_phase_safe(
            PhaseMetric(
                phase_name=phase_name,
                run_id=self.metrics_collector.run_id,
                episode_id=episode_id,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
            )
        )

    def execute_python_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code inside the configured Docker container."""
        if (
            self.docker_client is None
            or self.execution_container is None
            or self.code_host_cache_folder is None
        ):
            return {
                "success": False,
                "stdout": "",
                "stderr": "Docker execution context is not initialized",
                "return_code": -1,
                "executed_code": code,
            }

        in_container_script_path: Optional[str] = None
        reflected_code = code

        try:
            in_container_script_path, reflected_code = write_code_in_con(
                self.docker_client,
                self.execution_container,
                host_cache_folder=self.code_host_cache_folder,
                code=code,
                postfix="mode",
                in_container_path="/tmp",
            )
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Failed to write code to container: {e}",
                "return_code": -1,
                "executed_code": code,
            }

        try:
            execution_result = run_code_in_con(
                self.execution_container,
                in_container_script_path,
                timeout_seconds=timeout,
            )
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Container execution failed: {e}",
                "return_code": -1,
                "executed_code": reflected_code,
            }
        finally:
            if in_container_script_path is not None:
                try:
                    self.execution_container.exec_run(
                        cmd=["/bin/sh", "-c", f"rm -f {in_container_script_path}"],
                    )
                except Exception:
                    pass

        match execution_result:
            case Ok(execution_output):
                return {
                    "success": True,
                    "stdout": execution_output.strip(),
                    "stderr": "",
                    "return_code": 0,
                    "executed_code": reflected_code,
                }
            case Err(error_message):
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": error_message,
                    "return_code": -1,
                    "executed_code": reflected_code,
                }

        return {
            "success": False,
            "stdout": "",
            "stderr": "Unexpected container execution result",
            "return_code": -1,
            "executed_code": reflected_code,
        }

    def start_episode(self, episode_id: str):
        """Initialize new episode."""
        self.episode_state = EpisodeState(
            episode_id=episode_id, current_step=1, mode_history=[]
        )

        # Add episode start to scratchpad
        start_message = f"Started with budget {self.budget.total_budget}"
        self.scratchpad.append(start_message, episode_id)

        logger.info(f"Started episode {episode_id} with unified mode system")

    def can_take_action(self) -> bool:
        """Check if orchestrator can make more decisions."""
        return not self.budget.is_exhausted()

    def execute_orchestrator_mode(self) -> OrchestratorDecision:
        """Execute orchestrator mode - strategic planning and delegation."""
        if self.budget.is_exhausted():
            raise RuntimeError("Budget exhausted - cannot execute orchestrator mode")

        # Import here to avoid circular imports
        from src.mode_prompts import get_orchestrator_prompt_data
        from src.mode_parsers import parse_orchestrator_response

        # Build orchestrator prompt
        prompt_data = get_orchestrator_prompt_data(
            scratchpad_content=self.scratchpad.get_content(),
            budget_remaining=self.budget.remaining,
            total_budget=self.budget.total_budget,
        )

        messages = [
            self._message("system", prompt_data["system_prompt"], "orchestrator"),
            self._message("user", prompt_data["user_prompt"], "orchestrator"),
        ]

        started_at = time.perf_counter()
        match self.genner.plist_completion(messages):
            case Ok(inference_result):
                raw_response = inference_result.content
                # Print orchestrator conversation
                print(f"\n{'=' * 60}")
                print(f"🎯 ORCHESTRATOR MODE - Strategic Decision:")
                print(f"{'=' * 60}")
                print(f"BUDGET: {self.budget.remaining}/{self.budget.total_budget}")
                print(f"SCRATCHPAD: {self.scratchpad.get_content()[-300:]}...")
                print(f"\nCLAUDE DECISION:\n{raw_response}")
                print(f"{'=' * 60}\n")

                # Parse orchestrator response
                decision = parse_orchestrator_response(raw_response)
                decision.duration_ms = (time.perf_counter() - started_at) * 1000

                # Consume budget for orchestrator decision
                self.budget.consume(1)
                self._record_phase_metric(
                    "orchestrator",
                    decision.duration_ms,
                    True,
                )

                return decision

            case Err(error):
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._record_phase_metric(
                    "orchestrator", duration_ms, False, str(error)
                )
                logger.error(f"Orchestrator mode failed: {error}")
                raise RuntimeError(f"Orchestrator execution failed: {error}")

    def execute_mode(self, mode: Mode, instruction: str) -> ModeResult:
        """Execute specific mode with given instruction."""
        if mode == Mode.ORCHESTRATOR:
            raise ValueError("Use execute_orchestrator_mode() for orchestrator mode")

        logger.info(f"Executing {mode.value} mode: {instruction[:100]}...")

        started_at = time.perf_counter()
        if mode == Mode.INVESTIGATOR:
            result = self._execute_investigator_mode(instruction)
        elif mode == Mode.EXPLORER:
            result = self._execute_explorer_mode(instruction)
        elif mode == Mode.RECORDER:
            result = self._execute_recorder_mode(instruction)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        result.duration_ms = (time.perf_counter() - started_at) * 1000
        self._record_phase_metric(
            mode.value,
            result.duration_ms,
            result.success,
            result.error_message,
        )
        return result

    def _execute_investigator_mode(self, instruction: str) -> ModeResult:
        """Execute investigator mode - environment reconnaissance."""
        from src.mode_prompts import get_investigator_prompt_data
        from src.mode_parsers import parse_investigator_response

        prompt_data = get_investigator_prompt_data(
            instruction=instruction, scratchpad_content=self.scratchpad.get_content()
        )

        messages = [
            self._message("system", prompt_data["system_prompt"], "investigator"),
            self._message("user", prompt_data["user_prompt"], "investigator"),
        ]

        match self.genner.plist_completion(messages):
            case Ok(inference_result):
                raw_response = inference_result.content
                # Print investigator conversation
                print(f"\n{'=' * 60}")
                print(f"🔍 INVESTIGATOR MODE - Environment Analysis:")
                print(f"{'=' * 60}")
                print(f"INSTRUCTION: {instruction}")
                print(f"\nCLAUDE RESPONSE:\n{raw_response}")
                print(f"{'=' * 60}\n")

                # Parse response and create result
                parsed = parse_investigator_response(raw_response)

                # Execute the Python code if found
                execution_result = None
                if parsed.get("code"):
                    print(f"\n🔧 EXECUTING INVESTIGATOR CODE:")
                    print(f"{'=' * 60}")
                    print(parsed["code"])
                    print(f"{'=' * 60}")

                    execution_result = self.execute_python_code(parsed["code"])

                    print(f"\n📋 EXECUTION RESULTS:")
                    print(f"Success: {execution_result['success']}")
                    if execution_result["stdout"]:
                        print(f"STDOUT:\n{execution_result['stdout']}")
                    if execution_result["stderr"]:
                        print(f"STDERR:\n{execution_result['stderr']}")
                    print(f"{'=' * 60}\n")

                # Combine findings with execution results
                combined_findings = parsed["findings"]
                if execution_result:
                    if execution_result["success"] and execution_result["stdout"]:
                        combined_findings += (
                            f"\n\nExecution Output:\n{execution_result['stdout']}"
                        )
                    elif not execution_result["success"]:
                        combined_findings += (
                            f"\n\nExecution Error:\n{execution_result['stderr']}"
                        )

                # Auto-capture key discoveries in scratchpad (hybrid strategy)
                if combined_findings:
                    key_findings = (
                        combined_findings[:150] + "..."
                        if len(combined_findings) > 150
                        else combined_findings
                    )
                    self.scratchpad.append(f"[Investigator] {key_findings}")

                # Update structured data with execution results
                if execution_result:
                    parsed["execution_result"] = execution_result

                # Record interaction for training data
                prompt_response_pair = PromptResponsePair(
                    prompt=messages[1]["content"],
                    raw_response=raw_response,
                    timestamp=datetime.now().isoformat(),
                    interaction_type="investigator",
                    success=True,
                )

                return ModeResult(
                    mode=Mode.INVESTIGATOR,
                    success=True,
                    content=combined_findings,
                    structured_data=parsed,
                    code_executed=parsed.get("code"),
                    prompt_response_pair=prompt_response_pair,
                )

            case Err(error):
                logger.error(f"Investigator mode failed: {error}")
                return ModeResult(
                    mode=Mode.INVESTIGATOR,
                    success=False,
                    content=f"Investigation failed: {error}",
                    structured_data={},
                    error_message=str(error),
                )

    def _execute_explorer_mode(self, instruction: str) -> ModeResult:
        """Execute explorer mode - cleanup code execution."""
        from src.mode_prompts import get_explorer_prompt_data
        from src.mode_parsers import parse_explorer_response

        prompt_data = get_explorer_prompt_data(
            instruction=instruction, scratchpad_content=self.scratchpad.get_content()
        )

        messages = [
            self._message("system", prompt_data["system_prompt"], "explorer"),
            self._message("user", prompt_data["user_prompt"], "explorer"),
        ]

        match self.genner.plist_completion(messages):
            case Ok(inference_result):
                raw_response = inference_result.content
                # Print explorer conversation
                print(f"\n{'=' * 60}")
                print(f"💻 EXPLORER MODE - Cleanup Execution:")
                print(f"{'=' * 60}")
                print(f"INSTRUCTION: {instruction}")
                print(f"\nCLAUDE RESPONSE:\n{raw_response}")
                print(f"{'=' * 60}\n")

                # Parse response
                parsed = parse_explorer_response(raw_response)

                # Execute the Python cleanup code if found
                execution_result = None
                if parsed.get("code"):
                    print(f"\n🧹 EXECUTING EXPLORER CLEANUP CODE:")
                    print(f"{'=' * 60}")
                    print(parsed["code"])
                    print(f"{'=' * 60}")

                    execution_result = self.execute_python_code(
                        parsed["code"], timeout=90
                    )  # Longer timeout for cleanup operations

                    print(f"\n🏁 CLEANUP EXECUTION RESULTS:")
                    print(f"Success: {execution_result['success']}")
                    if execution_result["stdout"]:
                        print(f"STDOUT:\n{execution_result['stdout']}")
                    if execution_result["stderr"]:
                        print(f"STDERR:\n{execution_result['stderr']}")
                    print(f"{'=' * 60}\n")

                # Combine results with execution output
                combined_results = parsed["results"]
                if execution_result:
                    if execution_result["success"] and execution_result["stdout"]:
                        combined_results += (
                            f"\n\nExecution Output:\n{execution_result['stdout']}"
                        )
                    elif not execution_result["success"]:
                        combined_results += (
                            f"\n\nExecution Error:\n{execution_result['stderr']}"
                        )

                # Try to extract space freed from actual execution output
                space_freed = parsed.get("space_freed_kb", 0.0)
                if (
                    execution_result
                    and execution_result["success"]
                    and execution_result["stdout"]
                ):
                    # Look for space indications in execution output
                    stdout_text = execution_result["stdout"]
                    # Look for patterns like "freed 1024 KB" or "deleted 2.5 MB"
                    space_matches = re.findall(
                        r"(\d+\.?\d*)\s*(KB|MB|GB|bytes)", stdout_text, re.IGNORECASE
                    )
                    if space_matches:
                        # Convert to KB and update space_freed
                        for amount, unit in space_matches:
                            amount = float(amount)
                            unit = unit.upper()
                            if unit == "MB":
                                amount *= 1024
                            elif unit == "GB":
                                amount *= 1024 * 1024
                            elif unit == "BYTES":
                                amount /= 1024
                            space_freed = max(
                                space_freed, amount
                            )  # Take the largest value found

                # Update episode state with space freed
                if space_freed > 0:
                    assert self.episode_state is not None
                    self.episode_state.total_space_freed += space_freed

                # Auto-capture key results in scratchpad (hybrid strategy)
                if space_freed > 0:
                    self.scratchpad.append(
                        f"[Explorer] Freed {space_freed:.1f}KB: {combined_results[:100]}..."
                    )
                elif combined_results:
                    result_summary = (
                        combined_results[:120] + "..."
                        if len(combined_results) > 120
                        else combined_results
                    )
                    self.scratchpad.append(f"[Explorer] {result_summary}")

                # Update structured data with execution results and corrected space freed
                if execution_result:
                    parsed["execution_result"] = execution_result
                parsed["space_freed_kb"] = space_freed

                # Record interaction for training data
                prompt_response_pair = PromptResponsePair(
                    prompt=messages[1]["content"],
                    raw_response=raw_response,
                    timestamp=datetime.now().isoformat(),
                    interaction_type="explorer",
                    success=True,
                )

                return ModeResult(
                    mode=Mode.EXPLORER,
                    success=True,
                    content=combined_results,
                    structured_data=parsed,
                    code_executed=parsed.get("code"),
                    prompt_response_pair=prompt_response_pair,
                )

            case Err(error):
                logger.error(f"Explorer mode failed: {error}")
                return ModeResult(
                    mode=Mode.EXPLORER,
                    success=False,
                    content=f"Cleanup failed: {error}",
                    structured_data={},
                    error_message=str(error),
                )

    def _execute_recorder_mode(self, instruction: str) -> ModeResult:
        """Execute recorder mode - memory and scratchpad management."""
        from src.mode_prompts import get_recorder_prompt_data
        from src.mode_parsers import parse_recorder_response

        prompt_data = get_recorder_prompt_data(
            instruction=instruction, scratchpad_content=self.scratchpad.get_content()
        )

        messages = [
            self._message("system", prompt_data["system_prompt"], "recorder"),
            self._message("user", prompt_data["user_prompt"], "recorder"),
        ]

        match self.genner.plist_completion(messages):
            case Ok(inference_result):
                raw_response = inference_result.content
                # Print recorder conversation
                print(f"\n{'=' * 60}")
                print(f"📝 RECORDER MODE - Memory Management:")
                print(f"{'=' * 60}")
                print(f"INSTRUCTION: {instruction}")
                print(f"\nCLAUDE RESPONSE:\n{raw_response}")
                print(f"{'=' * 60}\n")

                # Parse response and update scratchpad
                parsed = parse_recorder_response(raw_response)

                # Update scratchpad with new information
                if "scratchpad_entry" in parsed:
                    self.scratchpad.append(parsed["scratchpad_entry"])

                # Record interaction for training data
                prompt_response_pair = PromptResponsePair(
                    prompt=messages[1]["content"],
                    raw_response=raw_response,
                    timestamp=datetime.now().isoformat(),
                    interaction_type="recorder",
                    success=True,
                )

                return ModeResult(
                    mode=Mode.RECORDER,
                    success=True,
                    content=parsed.get("confirmation", "Scratchpad updated"),
                    structured_data=parsed,
                    prompt_response_pair=prompt_response_pair,
                )

            case Err(error):
                logger.error(f"Recorder mode failed: {error}")
                return ModeResult(
                    mode=Mode.RECORDER,
                    success=False,
                    content=f"Recording failed: {error}",
                    structured_data={},
                    error_message=str(error),
                )

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get complete episode summary for training data."""
        assert self.episode_state is not None
        return {
            "episode_id": self.episode_state.episode_id,
            "total_steps": self.episode_state.current_step - 1,
            "budget_used": self.budget.total_budget - self.budget.remaining,
            "budget_remaining": self.budget.remaining,
            "total_space_freed": self.episode_state.total_space_freed,
            "mode_history": self.episode_state.mode_history,
            "scratchpad_final": self.scratchpad.get_content(),
            "success": self.episode_state.total_space_freed > 0.0,
        }
