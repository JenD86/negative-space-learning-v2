from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    EXPLORE = "explore"      # Environment reconnaissance
    STRATEGISE = "strategise" # Planning and reasoning  
    CODE = "code"            # Write/modify execution code
    DEBUG = "debug"          # Error analysis and information harvesting


@dataclass
class PromptResponsePair:
    """Records individual LLM prompt/response interaction."""
    prompt: str
    raw_response: str
    timestamp: str
    interaction_type: str  # "action_selection", "explore", "strategise", "code", "debug", "regenerate"
    success: bool
    error_message: Optional[str] = None


@dataclass
class ActionResult:
    success: bool
    observation: str
    scratchpad_update: str
    budget_consumed: int = 1
    error_message: Optional[str] = None
    # NEW: Record all LLM interactions within this action
    prompt_response_pairs: list[PromptResponsePair] = None
    
    def __post_init__(self):
        if self.prompt_response_pairs is None:
            self.prompt_response_pairs = []


class ActionBudget:
    def __init__(self, total_budget: int = 12):
        self.total_budget = total_budget
        self.remaining = total_budget
    
    def consume(self, amount: int = 1) -> bool:
        """Consume budget. Returns True if successful, False if insufficient."""
        if self.remaining >= amount:
            self.remaining -= amount
            return True
        return False
    
    def is_exhausted(self) -> bool:
        return self.remaining <= 0
    
    def reset(self, new_budget: Optional[int] = None):
        """Reset budget for new episode."""
        if new_budget is not None:
            self.total_budget = new_budget
        self.remaining = self.total_budget
