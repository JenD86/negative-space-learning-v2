from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime

from src.action_system import ActionType, ActionResult, ActionBudget
from src.scratchpad import CrossEpisodeScratchpad
from src.genner.Base import Genner
from src.typing.message import Message


@dataclass
class ActionExecution:
    """Record of a single action execution within an episode."""
    step: int
    action_type: ActionType
    budget_before: int
    result: ActionResult
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action_type": self.action_type.value,
            "budget_before": self.budget_before,
            "success": self.result.success,
            "observation": self.result.observation,
            "budget_consumed": self.result.budget_consumed,
            "error_message": self.result.error_message,
            "timestamp": self.timestamp
        }


class EpisodeController:
    def __init__(self, 
                 genner: Genner,
                 budget: int = 12,
                 scratchpad_max_chars: int = 10000,
                 scratchpad_path: Optional[Path] = None):
        self.genner = genner
        self.budget = ActionBudget(budget)
        self.scratchpad = CrossEpisodeScratchpad(scratchpad_max_chars, scratchpad_path)
        
        # Episode state
        self.episode_id: Optional[str] = None
        self.action_sequence: List[ActionExecution] = []
        self.step_count = 0
        self.completed = False
        self.space_freed_kb = 0.0
    
    def start_episode(self, episode_id: str) -> None:
        """Start a new episode."""
        self.episode_id = episode_id
        self.action_sequence = []
        self.step_count = 0
        self.completed = False
        self.space_freed_kb = 0.0
        self.budget.reset()
        
        # Add episode start to scratchpad
        self.scratchpad.append(f"Episode {episode_id} started, budget: {self.budget.total_budget}", episode_id)
    
    def can_take_action(self) -> bool:
        """Check if agent can take another action."""
        return not self.budget.is_exhausted() and not self.completed
    
    def get_agent_context(self) -> str:
        """Get current context for agent decision making."""
        context_parts = [
            f"Budget remaining: {self.budget.remaining}/{self.budget.total_budget}",
            f"Actions taken this episode: {self.step_count}",
        ]
        
        # Include recent scratchpad content (last 2000 chars to avoid overwhelming small models)
        scratchpad_content = self.scratchpad.get_content()
        if scratchpad_content:
            # Get recent content
            recent_content = scratchpad_content[-2000:] if len(scratchpad_content) > 2000 else scratchpad_content
            context_parts.append(f"\nRecent observations:\n{recent_content}")
        
        return "\n".join(context_parts)
    
    def execute_action(self, action_type: ActionType, context: Dict[str, Any]) -> ActionResult:
        """Execute a single action and record the result."""
        if not self.can_take_action():
            return ActionResult(
                success=False,
                observation="Cannot take action: budget exhausted or episode completed",
                scratchpad_update="[System] Budget exhausted or episode completed",
                error_message="Budget exhausted or episode completed"
            )
        
        budget_before = self.budget.remaining
        self.step_count += 1
        
        # Execute the action (placeholder - will be implemented with actual action handlers)
        result = self._execute_action_type(action_type, context)
        
        # Consume budget if action was attempted
        self.budget.consume(result.budget_consumed)
        
        # Record action execution
        execution = ActionExecution(
            step=self.step_count,
            action_type=action_type,
            budget_before=budget_before,
            result=result,
            timestamp=datetime.now().isoformat()
        )
        self.action_sequence.append(execution)
        
        # Automatically log key observations to scratchpad
        if result.observation:
            # Create compressed observation for scratchpad
            log_entry = f"{action_type.value.upper()}: {result.observation[:200]}"
            if result.error_message:
                log_entry += f" | Error: {result.error_message[:100]}"
            
            self.scratchpad.append(log_entry, self.episode_id)
        
        return result
    
    def _execute_action_type(self, action_type: ActionType, context: Dict[str, Any]) -> ActionResult:
        """Execute specific action type. Placeholder for actual implementation."""
        # This will be implemented with actual action handlers in the next steps
        return ActionResult(
            success=True,
            observation=f"Executed {action_type.value} action",
            scratchpad_update=f"[{action_type.value.title()}] Placeholder action executed",
            budget_consumed=1
        )
    
    def signal_completion(self, space_freed_kb: float) -> None:
        """Agent signals episode completion."""
        self.completed = True
        self.space_freed_kb = space_freed_kb
        self.scratchpad.append(
            f"Completed: {space_freed_kb}KB freed", 
            self.episode_id
        )
    
    def force_completion(self, reason: str = "Budget exhausted") -> None:
        """Force episode completion."""
        self.completed = True
        self.scratchpad.append(f"Ended: {reason}", self.episode_id)
    
    def is_successful(self) -> bool:
        """Check if episode was successful (ΔR > 0)."""
        return self.space_freed_kb > 0.0
    
    def get_trajectory(self) -> Dict[str, Any]:
        """Get complete episode trajectory for SFT training."""
        return {
            "episode_id": self.episode_id,
            "success": self.is_successful(),
            "space_freed_kb": self.space_freed_kb,
            "total_steps": len(self.action_sequence),
            "budget_used": self.budget.total_budget - self.budget.remaining,
            "action_sequence": [action.to_dict() for action in self.action_sequence],
            "scratchpad_content": self.scratchpad.get_content(),
            "completion_reason": "agent_signal" if self.completed else "budget_exhausted"
        }
