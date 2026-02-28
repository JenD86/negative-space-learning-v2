import re
from typing import Dict, Any, Optional
from loguru import logger

from src.mode_controller import OrchestratorDecision, Mode


def parse_orchestrator_response(raw_response: str) -> OrchestratorDecision:
    """Parse orchestrator response in MODE:/INSTRUCTION:/REASONING: format."""
    try:
        # Extract MODE
        mode_match = re.search(r'MODE:\s*(investigator|explorer|recorder)', raw_response, re.IGNORECASE)
        if not mode_match:
            logger.warning(f"No valid MODE found in orchestrator response: {raw_response[:100]}")
            # Default fallback to investigator
            target_mode = Mode.INVESTIGATOR
            mode_str = "investigator"
        else:
            mode_str = mode_match.group(1).lower()
            target_mode = Mode(mode_str)
        
        # Extract INSTRUCTION
        instruction_match = re.search(r'INSTRUCTION:\s*(.+?)(?=\nREASONING:|$)', raw_response, re.IGNORECASE | re.DOTALL)
        instruction = instruction_match.group(1).strip() if instruction_match else f"Default {mode_str} task"
        
        # Extract REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+?)$', raw_response, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return OrchestratorDecision(
            target_mode=target_mode,
            instruction=instruction,
            reasoning=reasoning,
            raw_response=raw_response
        )
        
    except Exception as e:
        logger.error(f"Failed to parse orchestrator response: {e}")
        # Fallback decision
        return OrchestratorDecision(
            target_mode=Mode.INVESTIGATOR,
            instruction="Analyze current system state",
            reasoning=f"Parser error, defaulting to investigation: {e}",
            raw_response=raw_response
        )


def parse_investigator_response(raw_response: str) -> Dict[str, Any]:
    """Parse investigator response to extract code and findings."""
    try:
        # Extract Python code block
        code_match = re.search(r'```python\s*\n(.*?)\n```', raw_response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        
        # Extract findings section
        findings_match = re.search(r'Findings:\s*(.+?)$', raw_response, re.IGNORECASE | re.DOTALL)
        findings = findings_match.group(1).strip() if findings_match else raw_response
        
        # Try to extract structured data from findings
        structured_data = _extract_investigator_data(findings)
        
        return {
            "code": code,
            "findings": findings,
            "structured_data": structured_data,
            "raw_response": raw_response
        }
        
    except Exception as e:
        logger.error(f"Failed to parse investigator response: {e}")
        return {
            "code": None,
            "findings": f"Parse error: {raw_response}",
            "structured_data": {},
            "raw_response": raw_response
        }


def parse_explorer_response(raw_response: str) -> Dict[str, Any]:
    """Parse explorer response to extract code, results, and space freed."""
    try:
        # Extract Python code block
        code_match = re.search(r'```python\s*\n(.*?)\n```', raw_response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        
        # Extract results section
        results_match = re.search(r'Results:\s*(.+?)$', raw_response, re.IGNORECASE | re.DOTALL)
        results = results_match.group(1).strip() if results_match else raw_response
        
        # Try to extract space freed from results
        space_freed = _extract_space_freed(results)
        
        # Extract other structured data
        structured_data = _extract_explorer_data(results)
        structured_data["space_freed_kb"] = space_freed
        
        return {
            "code": code,
            "results": results,
            "space_freed_kb": space_freed,
            "structured_data": structured_data,
            "raw_response": raw_response
        }
        
    except Exception as e:
        logger.error(f"Failed to parse explorer response: {e}")
        return {
            "code": None,
            "results": f"Parse error: {raw_response}",
            "space_freed_kb": 0.0,
            "structured_data": {},
            "raw_response": raw_response
        }


def parse_recorder_response(raw_response: str) -> Dict[str, Any]:
    """Parse recorder response to extract scratchpad updates."""
    try:
        # Extract updated information section
        update_match = re.search(r'Updated information:\s*(.+?)$', raw_response, re.IGNORECASE | re.DOTALL)
        update_content = update_match.group(1).strip() if update_match else raw_response
        
        # Create scratchpad entry with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        scratchpad_entry = f"[{timestamp}] {update_content}"
        
        return {
            "scratchpad_entry": scratchpad_entry,
            "update_content": update_content,
            "confirmation": f"Scratchpad updated: {update_content[:50]}...",
            "raw_response": raw_response
        }
        
    except Exception as e:
        logger.error(f"Failed to parse recorder response: {e}")
        return {
            "scratchpad_entry": f"[Parse Error] {raw_response}",
            "update_content": raw_response,
            "confirmation": "Scratchpad updated with raw response due to parse error",
            "raw_response": raw_response
        }


def _extract_investigator_data(findings: str) -> Dict[str, Any]:
    """Extract structured data from investigator findings."""
    structured = {}
    
    # Look for file sizes
    size_matches = re.findall(r'(\d+\.?\d*)\s*(KB|MB|GB)', findings, re.IGNORECASE)
    if size_matches:
        structured["sizes_found"] = [(float(size), unit.upper()) for size, unit in size_matches]
    
    # Look for file paths
    path_matches = re.findall(r'(/[^\s]+)', findings)
    if path_matches:
        structured["paths_found"] = path_matches
    
    # Look for numbers (counts, etc.)
    number_matches = re.findall(r'\b(\d+)\s+(files?|directories|processes)', findings, re.IGNORECASE)
    if number_matches:
        structured["counts"] = [(int(num), item) for num, item in number_matches]
    
    return structured


def _extract_explorer_data(results: str) -> Dict[str, Any]:
    """Extract structured data from explorer results."""
    structured = {}
    
    # Look for success/failure indicators
    if re.search(r'\b(success|completed|done|freed)\b', results, re.IGNORECASE):
        structured["success_indicators"] = True
    elif re.search(r'\b(failed|error|exception)\b', results, re.IGNORECASE):
        structured["success_indicators"] = False
    
    # Look for file operations
    if re.search(r'\b(deleted|removed|cleaned)\b', results, re.IGNORECASE):
        structured["operation_type"] = "deletion"
    elif re.search(r'\b(compressed|archived)\b', results, re.IGNORECASE):
        structured["operation_type"] = "compression"
    
    return structured


def _extract_space_freed(results: str) -> float:
    """Extract space freed in KB from explorer results."""
    # Look for explicit KB mentions
    kb_match = re.search(r'(\d+\.?\d*)\s*KB', results, re.IGNORECASE)
    if kb_match:
        return float(kb_match.group(1))
    
    # Look for MB mentions and convert to KB
    mb_match = re.search(r'(\d+\.?\d*)\s*MB', results, re.IGNORECASE)
    if mb_match:
        return float(mb_match.group(1)) * 1024
    
    # Look for GB mentions and convert to KB  
    gb_match = re.search(r'(\d+\.?\d*)\s*GB', results, re.IGNORECASE)
    if gb_match:
        return float(gb_match.group(1)) * 1024 * 1024
    
    # Look for generic "freed X" patterns
    freed_match = re.search(r'freed\s+(\d+\.?\d*)', results, re.IGNORECASE)
    if freed_match:
        # Assume KB if no unit specified
        return float(freed_match.group(1))
    
    return 0.0
