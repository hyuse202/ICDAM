"""
PSPLib Benchmark Data Loader
ICDAM 2025 - Table 2: Benchmark Data Infrastructure

This module implements parsing for standard PSPLib (.sm) format files used in
Resource-Constrained Project Scheduling Problem (RCPSP) benchmarks.

Reference:
- PSPLib: https://www.om-db.wi.tum.de/psplib/
- ICDAM 2025: Benchmark-centric Hybrid LLM Multi-Agent System for SCM
"""

import os
from typing import Dict, List, Any, Optional


def parse_psplib(file_path: str) -> Dict[str, Any]:
    """
    Parse a PSPLib .sm format file and extract project scheduling data.
    
    ICDAM 2025 Table 2: Benchmark Loader Implementation.
    
    This function extracts:
    - PRECEDENCE RELATIONS: Successor jobs for each task
    - REQUESTS/DURATIONS: Duration and resource demands (R1-R4) per job
    - RESOURCEAVAILABILITIES: Total capacities for renewable resources
    
    Args:
        file_path: Absolute or relative path to the .sm file.
        
    Returns:
        Dictionary with structure:
        {
            'metadata': {
                'file': str,
                'jobs': int,
                'resources': int,
                'horizon': int
            },
            'resources': {
                'R1': int, 'R2': int, 'R3': int, 'R4': int
            },
            'jobs': {
                job_id: {
                    'duration': int,
                    'demands': {'R1': int, 'R2': int, 'R3': int, 'R4': int},
                    'successors': List[int]
                }
            }
        }
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is invalid or cannot be parsed.
    """
    # Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PSPLib file not found: {file_path}")
    
    # Initialize result structure
    result: Dict[str, Any] = {
        'metadata': {
            'file': os.path.basename(file_path),
            'jobs': 0,
            'resources': 0,
            'horizon': 0
        },
        'resources': {},
        'jobs': {}
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        raise ValueError(f"Failed to read file {file_path}: {e}")
    
    # Parse state machine
    current_section: Optional[str] = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and delimiter lines
        if not line or line.startswith('*') or line.startswith('-'):
            continue
        
        # Detect section headers
        if line.startswith('PRECEDENCE RELATIONS:'):
            current_section = 'PRECEDENCE'
            continue
        elif line.startswith('REQUESTS/DURATIONS:'):
            current_section = 'REQUESTS'
            continue
        elif line.startswith('RESOURCEAVAILABILITIES:'):
            current_section = 'RESOURCES'
            continue
        elif line.startswith('PROJECT INFORMATION:'):
            current_section = 'PROJECT_INFO'
            continue
        elif ':' in line and current_section is None:
            # Parse metadata lines (e.g., "jobs (incl. supersource/sink ):  32")
            _parse_metadata_line(line, result)
            continue
        
        # Parse section content
        if current_section == 'PRECEDENCE':
            _parse_precedence_line(line, result)
        elif current_section == 'REQUESTS':
            _parse_requests_line(line, result)
        elif current_section == 'RESOURCES':
            _parse_resources_line(line, result)
    
    # Validate parsing result
    if not result['jobs']:
        raise ValueError(f"No jobs parsed from file: {file_path}")
    
    if not result['resources']:
        raise ValueError(f"No resource capacities parsed from file: {file_path}")
    
    return result


def _parse_metadata_line(line: str, result: Dict[str, Any]) -> None:
    """
    Parse metadata lines from PSPLib header.
    
    Args:
        line: Raw line from file.
        result: Result dictionary to update.
    """
    line_lower = line.lower()
    
    try:
        if 'jobs' in line_lower and 'supersource' in line_lower:
            # Extract total jobs count
            parts = line.split(':')
            if len(parts) >= 2:
                result['metadata']['jobs'] = int(parts[-1].strip())
        elif 'horizon' in line_lower:
            parts = line.split(':')
            if len(parts) >= 2:
                result['metadata']['horizon'] = int(parts[-1].strip())
        elif 'renewable' in line_lower:
            # Extract renewable resource count
            parts = line.split(':')
            if len(parts) >= 2:
                # Format: "  - renewable                 :  4   R"
                value_part = parts[-1].strip().split()[0]
                result['metadata']['resources'] = int(value_part)
    except (ValueError, IndexError):
        # Skip malformed metadata lines
        pass


def _parse_precedence_line(line: str, result: Dict[str, Any]) -> None:
    """
    Parse a line from PRECEDENCE RELATIONS section.
    
    Format: jobnr.    #modes  #successors   successors
    Example: 1        1          3           2   3   4
    
    Args:
        line: Raw line from precedence section.
        result: Result dictionary to update.
    """
    # Skip header line
    if line.startswith('jobnr'):
        return
    
    parts = line.split()
    if len(parts) < 3:
        return
    
    try:
        job_id = int(parts[0])
        num_successors = int(parts[2])
        
        # Extract successor job IDs (positions 3 onwards)
        successors: List[int] = []
        if num_successors > 0 and len(parts) > 3:
            for i in range(3, min(3 + num_successors, len(parts))):
                successors.append(int(parts[i]))
        
        # Initialize job entry if not exists
        if job_id not in result['jobs']:
            result['jobs'][job_id] = {
                'duration': 0,
                'demands': {},
                'successors': []
            }
        
        result['jobs'][job_id]['successors'] = successors
        
    except (ValueError, IndexError):
        # Skip malformed lines
        pass


def _parse_requests_line(line: str, result: Dict[str, Any]) -> None:
    """
    Parse a line from REQUESTS/DURATIONS section.
    
    Format: jobnr. mode duration  R 1  R 2  R 3  R 4
    Example: 2      1     2       1    2    4    0
    
    Args:
        line: Raw line from requests section.
        result: Result dictionary to update.
    """
    # Skip header line
    if line.startswith('jobnr'):
        return
    
    parts = line.split()
    if len(parts) < 7:
        return
    
    try:
        job_id = int(parts[0])
        # mode = int(parts[1])  # Currently unused (single-mode RCPSP)
        duration = int(parts[2])
        
        # Extract resource demands (R1, R2, R3, R4)
        demands = {
            'R1': int(parts[3]),
            'R2': int(parts[4]),
            'R3': int(parts[5]),
            'R4': int(parts[6])
        }
        
        # Initialize job entry if not exists
        if job_id not in result['jobs']:
            result['jobs'][job_id] = {
                'duration': 0,
                'demands': {},
                'successors': []
            }
        
        result['jobs'][job_id]['duration'] = duration
        result['jobs'][job_id]['demands'] = demands
        
    except (ValueError, IndexError):
        # Skip malformed lines
        pass


def _parse_resources_line(line: str, result: Dict[str, Any]) -> None:
    """
    Parse a line from RESOURCEAVAILABILITIES section.
    
    Format:
      R 1  R 2  R 3  R 4   (header)
       24   23   25   33   (values)
    
    Args:
        line: Raw line from resources section.
        result: Result dictionary to update.
    """
    # Skip header line (contains 'R')
    if 'R' in line:
        return
    
    parts = line.split()
    if len(parts) < 4:
        return
    
    try:
        result['resources'] = {
            'R1': int(parts[0]),
            'R2': int(parts[1]),
            'R3': int(parts[2]),
            'R4': int(parts[3])
        }
    except (ValueError, IndexError):
        # Skip malformed lines
        pass


def create_dummy_sm_file(file_path: str) -> None:
    """
    Create a minimal dummy PSPLib .sm file for testing purposes.
    
    ICDAM 2025: Ensures simulation can run without external data downloads.
    
    Args:
        file_path: Path where the dummy file will be created.
    """
    dummy_content = """************************************************************************
file with basedata            : dummy_test.bas
initial value random generator: 12345
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  5
horizon                       :  20
RESOURCES
  - renewable                 :  4   R
  - nonrenewable              :  0   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1      3      0       15        5       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          2           2   3
   2        1          1           4
   3        1          1           4
   4        1          1           5
   5        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       2    1    3    1
  3      1     4       1    2    1    2
  4      1     5       3    2    2    1
  5      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4
    5    4    5    4
************************************************************************
"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(dummy_content)
    
    print(f"[PSPLib Loader] Created dummy file: {file_path}")


def print_parsed_data(data: Dict[str, Any]) -> None:
    """
    Pretty-print parsed PSPLib data for debugging.
    
    Args:
        data: Parsed data dictionary from parse_psplib().
    """
    print("\n" + "=" * 60)
    print("PARSED PSPLIB DATA")
    print("=" * 60)
    
    print(f"\n[Metadata]")
    print(f"  File: {data['metadata']['file']}")
    print(f"  Jobs: {data['metadata']['jobs']}")
    print(f"  Resources: {data['metadata']['resources']}")
    print(f"  Horizon: {data['metadata']['horizon']}")
    
    print(f"\n[Resource Capacities]")
    for res, cap in data['resources'].items():
        print(f"  {res}: {cap}")
    
    print(f"\n[Jobs] (showing first 5)")
    job_ids = sorted(data['jobs'].keys())[:5]
    for job_id in job_ids:
        job = data['jobs'][job_id]
        print(f"  Job {job_id}:")
        print(f"    Duration: {job['duration']}")
        print(f"    Demands: {job['demands']}")
        print(f"    Successors: {job['successors']}")
    
    if len(data['jobs']) > 5:
        print(f"  ... and {len(data['jobs']) - 5} more jobs")
    
    print("=" * 60)


# Module test
if __name__ == "__main__":
    import sys
    
    print("PSPLib Loader - ICDAM 2025 Table 2 Implementation")
    print("-" * 50)
    
    # Test with actual file or create dummy #test với 1 file đầu tiên 
    test_file = os.path.join("data", "raw", "rcpsp", "j30", "j3010_1.sm")
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Creating dummy file for testing...")
        create_dummy_sm_file("data/raw/rcpsp/test/dummy_test.sm")
        test_file = "data/raw/rcpsp/test/dummy_test.sm"
    
    try:
        data = parse_psplib(test_file)
        print_parsed_data(data)
        print("\n[SUCCESS] PSPLib loader working correctly.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
