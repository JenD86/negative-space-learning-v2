from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from enum import Enum
from loguru import logger
from result import Ok, Err
import subprocess
import tempfile
import os
import re

from src.action_system import ActionBudget, PromptResponsePair
from src.scratchpad import CrossEpisodeScratchpad
from src.genner.Base import Genner
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


@dataclass
class ModeResult:
    """Standardized result format across all modes."""
    mode: Mode
    success: bool
    content: str  # Main response content
    structured_data: Dict[str, Any]  # Parsed data (findings, space_freed, etc.)
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
    
    def add_mode_execution(self, mode: Mode, instruction: str, result: ModeResult):
        """Record a mode execution in history."""
        self.mode_history.append({
            "step": self.current_step,
            "mode": mode.value,
            "instruction": instruction,
            "success": result.success,
            "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
            "timestamp": datetime.now().isoformat()
        })
        self.current_step += 1


def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code and return output, errors, and success status."""
    try:
        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name
        
        # Execute the Python code with timeout
        result = subprocess.run(
            ['python', temp_file], 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        # Clean up temporary file
        os.unlink(temp_file)
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "return_code": result.returncode,
            "executed_code": code
        }
        
    except subprocess.TimeoutExpired:
        # Clean up temporary file on timeout
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Code execution timed out after {timeout} seconds",
            "return_code": -1,
            "executed_code": code
        }
        
    except Exception as e:
        # Clean up temporary file on any other exception
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Code execution failed: {str(e)}",
            "return_code": -1,
            "executed_code": code
        }


class ModeController:
    """Unified AI system operating in different modes with single persona."""
    
    def __init__(self, genner: Genner, config: AppConfig):
        self.genner = genner
        self.config = config
        
        # Initialize episode components
        from pathlib import Path
        episode_config = config.episode or config.EpisodeConfig()
        self.budget = ActionBudget(total_budget=episode_config.action_budget)
        
        # Convert storage path to Path object if it's a string
        storage_path = episode_config.scratchpad_storage_path
        if storage_path and isinstance(storage_path, str):
            storage_path = Path(storage_path)
            
        self.scratchpad = CrossEpisodeScratchpad(
            max_chars=episode_config.scratchpad_max_chars,
            storage_path=storage_path
        )
        
        # Episode state tracking
        self.episode_state = None
        self.current_mode = Mode.ORCHESTRATOR
        
    def start_episode(self, episode_id: str):
        """Initialize new episode."""
        self.episode_state = EpisodeState(
            episode_id=episode_id,
            current_step=1,
            mode_history=[]
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
            total_budget=self.budget.total_budget
        )
        
        messages = [
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]}
        ]
        
        # Execute with Claude
        match self.genner.plist_completion(messages):
            case Ok(raw_response):
                # Print orchestrator conversation
                print(f"\n{'='*60}")
                print(f"🎯 ORCHESTRATOR MODE - Strategic Decision:")
                print(f"{'='*60}")
                print(f"BUDGET: {self.budget.remaining}/{self.budget.total_budget}")
                print(f"SCRATCHPAD: {self.scratchpad.get_content()[-300:]}...")
                print(f"\nCLAUDE DECISION:\n{raw_response}")
                print(f"{'='*60}\n")
                
                # Parse orchestrator response
                decision = parse_orchestrator_response(raw_response)
                
                # Consume budget for orchestrator decision
                self.budget.consume(1)
                
                return decision
                
            case Err(error):
                logger.error(f"Orchestrator mode failed: {error}")
                raise RuntimeError(f"Orchestrator execution failed: {error}")
    
    def execute_mode(self, mode: Mode, instruction: str) -> ModeResult:
        """Execute specific mode with given instruction."""
        if mode == Mode.ORCHESTRATOR:
            raise ValueError("Use execute_orchestrator_mode() for orchestrator mode")
            
        logger.info(f"Executing {mode.value} mode: {instruction[:100]}...")
        
        # Import mode-specific functions
        if mode == Mode.INVESTIGATOR:
            return self._execute_investigator_mode(instruction)
        elif mode == Mode.EXPLORER:
            return self._execute_explorer_mode(instruction)
        elif mode == Mode.RECORDER:
            return self._execute_recorder_mode(instruction)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _execute_investigator_mode(self, instruction: str) -> ModeResult:
        """Execute investigator mode - environment reconnaissance."""
        from src.mode_prompts import get_investigator_prompt_data
        from src.mode_parsers import parse_investigator_response
        
        prompt_data = get_investigator_prompt_data(
            instruction=instruction,
            scratchpad_content=self.scratchpad.get_content()
        )
        
        messages = [
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]}
        ]
        
        match self.genner.plist_completion(messages):
            case Ok(raw_response):
                # Print investigator conversation
                print(f"\n{'='*60}")
                print(f"🔍 INVESTIGATOR MODE - Environment Analysis:")
                print(f"{'='*60}")
                print(f"INSTRUCTION: {instruction}")
                print(f"\nCLAUDE RESPONSE:\n{raw_response}")
                print(f"{'='*60}\n")
                
                # Parse response and create result
                parsed = parse_investigator_response(raw_response)
                
                # Execute the Python code if found
                execution_result = None
                if parsed.get("code"):
                    print(f"\n🔧 EXECUTING INVESTIGATOR CODE:")
                    print(f"{'='*60}")
                    print(parsed["code"])
                    print(f"{'='*60}")
                    
                    execution_result = execute_python_code(parsed["code"])
                    
                    print(f"\n📋 EXECUTION RESULTS:")
                    print(f"Success: {execution_result['success']}")
                    if execution_result["stdout"]:
                        print(f"STDOUT:\n{execution_result['stdout']}")
                    if execution_result["stderr"]:
                        print(f"STDERR:\n{execution_result['stderr']}")
                    print(f"{'='*60}\n")
                
                # Combine findings with execution results
                combined_findings = parsed["findings"]
                if execution_result:
                    if execution_result["success"] and execution_result["stdout"]:
                        combined_findings += f"\n\nExecution Output:\n{execution_result['stdout']}"
                    elif not execution_result["success"]:
                        combined_findings += f"\n\nExecution Error:\n{execution_result['stderr']}"
                
                # Auto-capture key discoveries in scratchpad (hybrid strategy)
                if combined_findings:
                    key_findings = combined_findings[:150] + "..." if len(combined_findings) > 150 else combined_findings
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
                    success=True
                )
                
                return ModeResult(
                    mode=Mode.INVESTIGATOR,
                    success=True,
                    content=combined_findings,
                    structured_data=parsed,
                    code_executed=parsed.get("code"),
                    prompt_response_pair=prompt_response_pair
                )
                
            case Err(error):
                logger.error(f"Investigator mode failed: {error}")
                return ModeResult(
                    mode=Mode.INVESTIGATOR,
                    success=False,
                    content=f"Investigation failed: {error}",
                    structured_data={},
                    error_message=str(error)
                )
    
    def _execute_explorer_mode(self, instruction: str) -> ModeResult:
        """Execute explorer mode - cleanup code execution.""" 
        from src.mode_prompts import get_explorer_prompt_data
        from src.mode_parsers import parse_explorer_response
        
        prompt_data = get_explorer_prompt_data(
            instruction=instruction,
            scratchpad_content=self.scratchpad.get_content()
        )
        
        messages = [
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]}
        ]
        
        match self.genner.plist_completion(messages):
            case Ok(raw_response):
                # Print explorer conversation
                print(f"\n{'='*60}")
                print(f"💻 EXPLORER MODE - Cleanup Execution:")
                print(f"{'='*60}")
                print(f"INSTRUCTION: {instruction}")
                print(f"\nCLAUDE RESPONSE:\n{raw_response}")
                print(f"{'='*60}\n")
                
                # Parse response
                parsed = parse_explorer_response(raw_response)
                
                # Execute the Python cleanup code if found
                execution_result = None
                if parsed.get("code"):
                    print(f"\n🧹 EXECUTING EXPLORER CLEANUP CODE:")
                    print(f"{'='*60}")
                    print(parsed["code"])
                    print(f"{'='*60}")
                    
                    execution_result = execute_python_code(parsed["code"], timeout=90)  # Longer timeout for cleanup operations
                    
                    print(f"\n🏁 CLEANUP EXECUTION RESULTS:")
                    print(f"Success: {execution_result['success']}")
                    if execution_result["stdout"]:
                        print(f"STDOUT:\n{execution_result['stdout']}")
                    if execution_result["stderr"]:
                        print(f"STDERR:\n{execution_result['stderr']}")
                    print(f"{'='*60}\n")
                
                # Combine results with execution output
                combined_results = parsed["results"]
                if execution_result:
                    if execution_result["success"] and execution_result["stdout"]:
                        combined_results += f"\n\nExecution Output:\n{execution_result['stdout']}"
                    elif not execution_result["success"]:
                        combined_results += f"\n\nExecution Error:\n{execution_result['stderr']}"
                
                # Try to extract space freed from actual execution output
                space_freed = parsed.get("space_freed_kb", 0.0)
                if execution_result and execution_result["success"] and execution_result["stdout"]:
                    # Look for space indications in execution output
                    stdout_text = execution_result["stdout"]
                    # Look for patterns like "freed 1024 KB" or "deleted 2.5 MB"
                    space_matches = re.findall(r'(\d+\.?\d*)\s*(KB|MB|GB|bytes)', stdout_text, re.IGNORECASE)
                    if space_matches:
                        # Convert to KB and update space_freed
                        for amount, unit in space_matches:
                            amount = float(amount)
                            unit = unit.upper()
                            if unit == 'MB':
                                amount *= 1024
                            elif unit == 'GB':
                                amount *= 1024 * 1024
                            elif unit == 'BYTES':
                                amount /= 1024
                            space_freed = max(space_freed, amount)  # Take the largest value found
                
                # Update episode state with space freed
                if space_freed > 0:
                    self.episode_state.total_space_freed += space_freed
                
                # Auto-capture key results in scratchpad (hybrid strategy)
                if space_freed > 0:
                    self.scratchpad.append(f"[Explorer] Freed {space_freed:.1f}KB: {combined_results[:100]}...")
                elif combined_results:
                    result_summary = combined_results[:120] + "..." if len(combined_results) > 120 else combined_results
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
                    success=True
                )
                
                return ModeResult(
                    mode=Mode.EXPLORER,
                    success=True,
                    content=combined_results,
                    structured_data=parsed,
                    code_executed=parsed.get("code"),
                    prompt_response_pair=prompt_response_pair
                )
                
            case Err(error):
                logger.error(f"Explorer mode failed: {error}")
                return ModeResult(
                    mode=Mode.EXPLORER,
                    success=False,
                    content=f"Cleanup failed: {error}",
                    structured_data={},
                    error_message=str(error)
                )
    
    def _execute_recorder_mode(self, instruction: str) -> ModeResult:
        """Execute recorder mode - memory and scratchpad management."""
        from src.mode_prompts import get_recorder_prompt_data
        from src.mode_parsers import parse_recorder_response
        
        prompt_data = get_recorder_prompt_data(
            instruction=instruction,
            scratchpad_content=self.scratchpad.get_content()
        )
        
        messages = [
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]}
        ]
        
        match self.genner.plist_completion(messages):
            case Ok(raw_response):
                # Print recorder conversation
                print(f"\n{'='*60}")
                print(f"📝 RECORDER MODE - Memory Management:")
                print(f"{'='*60}")
                print(f"INSTRUCTION: {instruction}")
                print(f"\nCLAUDE RESPONSE:\n{raw_response}")
                print(f"{'='*60}\n")
                
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
                    success=True
                )
                
                return ModeResult(
                    mode=Mode.RECORDER,
                    success=True,
                    content=parsed.get("confirmation", "Scratchpad updated"),
                    structured_data=parsed,
                    prompt_response_pair=prompt_response_pair
                )
                
            case Err(error):
                logger.error(f"Recorder mode failed: {error}")
                return ModeResult(
                    mode=Mode.RECORDER,
                    success=False,
                    content=f"Recording failed: {error}",
                    structured_data={},
                    error_message=str(error)
                )
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get complete episode summary for training data."""
        return {
            "episode_id": self.episode_state.episode_id,
            "total_steps": self.episode_state.current_step - 1,
            "budget_used": self.budget.total_budget - self.budget.remaining,
            "budget_remaining": self.budget.remaining,
            "total_space_freed": self.episode_state.total_space_freed,
            "mode_history": self.episode_state.mode_history,
            "scratchpad_final": self.scratchpad.get_content(),
            "success": self.episode_state.total_space_freed > 0.0
        }
