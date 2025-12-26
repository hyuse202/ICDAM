"""
Plan Parser Module - OptiMUS Pattern Implementation
ICDAM 2025 - Verification Layer: LLM Output → Structured Schedule

This module extracts structured schedule data from LLM's natural language output.
Supports multiple formats: JSON blocks, table format, and natural language.

References:
- OptiMUS: Scalable Optimization Modeling with LLM (NeurIPS 2023)
- ICDAM 2025: Hybrid LLM + Solver Architecture
"""

import re
import json
from typing import Dict, Optional, List, Tuple


def parse_schedule_from_text(text: str) -> Dict[int, int]:
    """
    Extract structured schedule data from LLM's natural language output.
    
    OptiMUS Pattern: Converts unstructured LLM response to solver-verifiable format.
    
    Supports multiple formats:
    1. JSON block: {"1": 0, "2": 3, "3": 5}
    2. Table format: Job 1: Start 0, Job 2: Start 3
    3. Natural language: "Job 1 starts at time 0, Job 2 begins at 3"
    
    Args:
        text: Raw text output from LLM containing schedule information.
        
    Returns:
        Dictionary mapping job_id (int) to start_time (int).
        Returns empty dict if parsing fails completely.
        
    Example:
        >>> text = "Schedule: Job 1: Start 0, Job 2: Start 3, Job 3: Start 5"
        >>> parse_schedule_from_text(text)
        {1: 0, 2: 3, 3: 5}
    """
    schedule: Dict[int, int] = {}
    
    # Strategy 1: Try to find and parse JSON block
    json_schedule = _extract_json_schedule(text)
    if json_schedule:
        return json_schedule
    
    # Strategy 2: Parse "Job X: Start Y" format
    table_schedule = _extract_table_format(text)
    if table_schedule:
        return table_schedule
    
    # Strategy 3: Parse natural language patterns
    nl_schedule = _extract_natural_language(text)
    if nl_schedule:
        return nl_schedule
    
    # Strategy 4: Parse simple "job_id: start_time" pairs
    simple_schedule = _extract_simple_pairs(text)
    if simple_schedule:
        return simple_schedule
    
    return schedule


def _extract_json_schedule(text: str) -> Optional[Dict[int, int]]:
    """
    Extract schedule from JSON block in text.
    
    Looks for patterns like:
    - ```json {...} ```
    - {...}
    - schedule = {...}
    
    Args:
        text: Raw text containing potential JSON.
        
    Returns:
        Parsed schedule dict or None if not found/invalid.
    """
    # Pattern 1: Markdown code block with json
    json_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    
    if not match:
        # Pattern 2: Standalone JSON object
        json_pattern = r'\{[^{}]*"?\d+"?\s*:\s*\d+[^{}]*\}'
        match = re.search(json_pattern, text)
    
    if match:
        try:
            json_str = match.group(1) if '```' in text else match.group(0)
            parsed = json.loads(json_str)
            
            # Convert string keys to int
            schedule = {}
            for key, value in parsed.items():
                job_id = int(key) if isinstance(key, str) else key
                start_time = int(value) if isinstance(value, str) else value
                schedule[job_id] = start_time
            
            return schedule if schedule else None
            
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
    
    return None


def _extract_table_format(text: str) -> Optional[Dict[int, int]]:
    """
    Extract schedule from table-like format.
    
    Patterns:
    - "Job 1: Start 0" or "Job 1: Start=0"
    - "Job 1 starts at 0"
    - "Task 1: 0"
    
    Args:
        text: Raw text containing table format.
        
    Returns:
        Parsed schedule dict or None if not found.
    """
    schedule: Dict[int, int] = {}
    
    # Pattern: "Job X: Start Y" or "Job X: Start=Y" or "Job X: Y"
    patterns = [
        r'[Jj]ob\s*(\d+)\s*:\s*[Ss]tart\s*[=:]?\s*(\d+)',
        r'[Jj]ob\s*(\d+)\s*[:\-]\s*(\d+)',
        r'[Tt]ask\s*(\d+)\s*:\s*[Ss]tart\s*[=:]?\s*(\d+)',
        r'[Tt]ask\s*(\d+)\s*[:\-]\s*(\d+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for job_str, start_str in matches:
                try:
                    job_id = int(job_str)
                    start_time = int(start_str)
                    schedule[job_id] = start_time
                except ValueError:
                    continue
    
    return schedule if schedule else None


def _extract_natural_language(text: str) -> Optional[Dict[int, int]]:
    """
    Extract schedule from natural language descriptions.
    
    Patterns:
    - "Job 1 starts at time 0"
    - "Job 2 begins at 3"
    - "Start Job 3 at time 5"
    
    Args:
        text: Raw text containing natural language schedule.
        
    Returns:
        Parsed schedule dict or None if not found.
    """
    schedule: Dict[int, int] = {}
    
    patterns = [
        r'[Jj]ob\s*(\d+)\s*(?:starts?|begins?)\s*(?:at)?\s*(?:time)?\s*(\d+)',
        r'[Ss]tart\s*[Jj]ob\s*(\d+)\s*(?:at)?\s*(?:time)?\s*(\d+)',
        r'[Tt]ask\s*(\d+)\s*(?:starts?|begins?)\s*(?:at)?\s*(?:time)?\s*(\d+)',
        r'(?:at\s*)?[Tt]ime\s*(\d+)\s*[:\-]?\s*[Jj]ob\s*(\d+)',  # Reversed: time then job
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                try:
                    # Handle reversed pattern (time, job)
                    if 'time' in pattern.lower() and pattern.index('[Tt]ime') < pattern.index('[Jj]ob'):
                        start_time, job_id = int(match[0]), int(match[1])
                    else:
                        job_id, start_time = int(match[0]), int(match[1])
                    schedule[job_id] = start_time
                except (ValueError, IndexError):
                    continue
    
    return schedule if schedule else None


def _extract_simple_pairs(text: str) -> Optional[Dict[int, int]]:
    """
    Extract schedule from simple "key: value" or "key = value" pairs.
    
    Last resort parsing for formats like:
    - "1: 0, 2: 3, 3: 5"
    - "1=0, 2=3, 3=5"
    
    Args:
        text: Raw text containing simple pairs.
        
    Returns:
        Parsed schedule dict or None if not found.
    """
    schedule: Dict[int, int] = {}
    
    # Pattern: number : number or number = number
    pattern = r'(\d+)\s*[=:]\s*(\d+)'
    matches = re.findall(pattern, text)
    
    if matches:
        for key_str, value_str in matches:
            try:
                job_id = int(key_str)
                start_time = int(value_str)
                # Sanity check: job_id should be reasonable (1-1000)
                if 1 <= job_id <= 1000 and start_time >= 0:
                    schedule[job_id] = start_time
            except ValueError:
                continue
    
    return schedule if schedule else None


def validate_schedule_completeness(
    schedule: Dict[int, int], 
    expected_jobs: List[int]
) -> Tuple[bool, List[int]]:
    """
    Check if schedule contains all expected jobs.
    
    Args:
        schedule: Parsed schedule dictionary.
        expected_jobs: List of job IDs that should be present.
        
    Returns:
        Tuple of (is_complete: bool, missing_jobs: List[int]).
    """
    missing = [job for job in expected_jobs if job not in schedule]
    return (len(missing) == 0, missing)


def format_schedule_for_display(schedule: Dict[int, int]) -> str:
    """
    Format schedule dictionary for human-readable display.
    
    Args:
        schedule: Schedule dictionary {job_id: start_time}.
        
    Returns:
        Formatted string representation.
    """
    if not schedule:
        return "Empty schedule"
    
    lines = ["Schedule:"]
    for job_id in sorted(schedule.keys()):
        start_time = schedule[job_id]
        lines.append(f"  Job {job_id}: Start at time {start_time}")
    
    return "\n".join(lines)


# =============================================================================
# Module Test
# =============================================================================

def main():
    """Test the plan parser with various input formats."""
    print("=" * 70)
    print("PLAN PARSER - OptiMUS Pattern Test")
    print("=" * 70)
    
    test_cases = [
        # Test 1: JSON format
        (
            "JSON Format",
            '''Here is the schedule:
            ```json
            {"1": 0, "2": 3, "3": 7, "4": 12}
            ```
            '''
        ),
        # Test 2: Table format
        (
            "Table Format",
            "Schedule: Job 1: Start 0, Job 2: Start 3, Job 3: Start 7"
        ),
        # Test 3: Natural language
        (
            "Natural Language",
            "Job 1 starts at time 0. Job 2 begins at 3. Job 3 starts at time 7."
        ),
        # Test 4: Simple pairs
        (
            "Simple Pairs",
            "Proposed schedule: 1=0, 2=3, 3=7, 4=12"
        ),
        # Test 5: Mixed/Complex
        (
            "Mixed Format",
            "Based on analysis, I propose: Job 1: Start 0 (dummy), Job 2: Start 0, Job 3: Start 3"
        ),
        # Test 6: Empty/Invalid
        (
            "Invalid Input",
            "I don't know the schedule."
        ),
    ]
    
    for idx, (name, text) in enumerate(test_cases, 1):
        print(f"\n[Test {idx}] {name}")
        print("-" * 50)
        print(f"Input: {text[:60]}...")
        
        result = parse_schedule_from_text(text)
        
        if result:
            print(f"✓ Parsed successfully: {result}")
        else:
            print("✗ Failed to parse (empty result)")
    
    print("\n" + "=" * 70)
    print("PARSER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
