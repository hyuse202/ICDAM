"""
RCPSP Solver using Google OR-Tools CP-SAT
Solves Resource-Constrained Project Scheduling Problem from PSPLib instances.

OptiMUS Pattern Extension: Includes verify_schedule() for LLM + Solver feedback loop.
"""

import os
import re
from typing import Dict, List, Any, Optional
from ortools.sat.python import cp_model


class RCPSPParser:
    """
    Parses PSPLib .sm files for RCPSP instances.
    Extracts job durations, resource requirements, precedence relations, and resource capacities.
    """
    
    def __init__(self, filepath):
        """
        Initialize parser with the path to a .sm file.
        
        Args:
            filepath: Path to the PSPLib .sm file
        """
        self.filepath = filepath
        self.n_jobs = 0
        self.n_resources = 0
        self.durations = []
        self.resource_requirements = []
        self.successors = []
        self.resource_capacities = []
        
    def parse(self):
        """
        Parse the .sm file and extract all relevant data.
        
        Returns:
            dict: Parsed data containing:
                - n_jobs: Number of jobs (including dummy start/end)
                - n_resources: Number of renewable resources
                - durations: List of job durations
                - resource_requirements: List of lists [R1, R2, R3, R4] for each job
                - successors: List of lists of successor job IDs (0-indexed)
                - resource_capacities: List of max capacities for each resource
        """
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        # Extract number of jobs
        jobs_match = re.search(r'jobs \(incl\. supersource/sink \):\s+(\d+)', content)
        if jobs_match:
            self.n_jobs = int(jobs_match.group(1))
        
        # Extract number of renewable resources
        resources_match = re.search(r'- renewable\s+:\s+(\d+)\s+R', content)
        if resources_match:
            self.n_resources = int(resources_match.group(1))
        
        # Parse PRECEDENCE RELATIONS
        self._parse_precedence_relations(content)
        
        # Parse REQUESTS/DURATIONS
        self._parse_requests_durations(content)
        
        # Parse RESOURCEAVAILABILITIES
        self._parse_resource_availabilities(content)
        
        return {
            'n_jobs': self.n_jobs,
            'n_resources': self.n_resources,
            'durations': self.durations,
            'resource_requirements': self.resource_requirements,
            'successors': self.successors,
            'resource_capacities': self.resource_capacities
        }
    
    def _parse_precedence_relations(self, content):
        """Parse the PRECEDENCE RELATIONS section."""
        # Find the section
        prec_start = content.find('PRECEDENCE RELATIONS:')
        if prec_start == -1:
            raise ValueError("PRECEDENCE RELATIONS section not found")
        
        # Find the end of the section (next asterisk line)
        prec_end = content.find('***', prec_start + 1)
        prec_section = content[prec_start:prec_end]
        
        # Initialize successors list
        self.successors = [[] for _ in range(self.n_jobs)]
        
        # Parse each line
        lines = prec_section.split('\n')
        for line in lines[2:]:  # Skip header lines
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            try:
                job_id = int(parts[0]) - 1  # Convert to 0-indexed
                n_successors = int(parts[2])
                
                if n_successors > 0 and len(parts) >= 3 + n_successors:
                    succ_list = [int(parts[3 + i]) - 1 for i in range(n_successors)]
                    self.successors[job_id] = succ_list
            except (ValueError, IndexError):
                continue
    
    def _parse_requests_durations(self, content):
        """Parse the REQUESTS/DURATIONS section."""
        # Find the section
        req_start = content.find('REQUESTS/DURATIONS:')
        if req_start == -1:
            raise ValueError("REQUESTS/DURATIONS section not found")
        
        # Find the end of the section
        req_end = content.find('***', req_start + 1)
        req_section = content[req_start:req_end]
        
        # Initialize lists
        self.durations = [0] * self.n_jobs
        self.resource_requirements = [[0] * self.n_resources for _ in range(self.n_jobs)]
        
        # Parse each line
        lines = req_section.split('\n')
        for line in lines[3:]:  # Skip header lines (3 lines)
            line = line.strip()
            if not line or line.startswith('-') or line.startswith('*'):
                continue
            
            parts = line.split()
            if len(parts) < 3 + self.n_resources:
                continue
            
            try:
                job_id = int(parts[0]) - 1  # Convert to 0-indexed
                duration = int(parts[2])
                resources = [int(parts[3 + i]) for i in range(self.n_resources)]
                
                self.durations[job_id] = duration
                self.resource_requirements[job_id] = resources
            except (ValueError, IndexError):
                continue
    
    def _parse_resource_availabilities(self, content):
        """Parse the RESOURCEAVAILABILITIES section."""
        # Find the section
        res_start = content.find('RESOURCEAVAILABILITIES:')
        if res_start == -1:
            raise ValueError("RESOURCEAVAILABILITIES section not found")
        
        # Find the capacity line (should be 2 lines after the header)
        res_section = content[res_start:]
        lines = res_section.split('\n')
        
        # The format is:
        # RESOURCEAVAILABILITIES:
        #   R 1  R 2  R 3  R 4
        #    12   13    4   12
        
        for line in lines[2:4]:  # Check lines 2-3 after header
            line = line.strip()
            if not line or line.startswith('R ') or line.startswith('*'):
                continue
            
            parts = line.split()
            if len(parts) >= self.n_resources:
                try:
                    self.resource_capacities = [int(parts[i]) for i in range(self.n_resources)]
                    break
                except ValueError:
                    continue


class RCPSPSolver:
    """
    Solves RCPSP instances using Google OR-Tools CP-SAT solver.
    """
    
    def __init__(self):
        """Initialize the solver."""
        self.model = None
        self.solver = None
        # Store decision variables for solution extraction
        self._starts = []
        self._ends = []
        self._intervals = []
        
    def solve(self, data, time_limit_seconds=300):
        """
        Solve the RCPSP instance.
        
        Args:
            data: Parsed data dictionary from RCPSPParser
            time_limit_seconds: Time limit for the solver (default: 300s)
            
        Returns:
            tuple: (makespan, status_string)
                - makespan: Optimal/best makespan found (project duration)
                - status_string: Solver status (OPTIMAL, FEASIBLE, INFEASIBLE, etc.)
        """
        n_jobs = data['n_jobs']
        n_resources = data['n_resources']
        durations = data['durations']
        resource_requirements = data['resource_requirements']
        successors = data['successors']
        resource_capacities = data['resource_capacities']
        
        # Create the model
        self.model = cp_model.CpModel()
        
        # Compute horizon (upper bound on makespan)
        horizon = sum(durations)
        
        # Decision variables - store in instance for later access
        self._starts = []
        self._ends = []
        self._intervals = []
        
        # Create interval variables for each job
        for job in range(n_jobs):
            start = self.model.NewIntVar(0, horizon, f'start_{job}')
            duration = durations[job]
            end = self.model.NewIntVar(0, horizon, f'end_{job}')
            interval = self.model.NewIntervalVar(start, duration, end, f'interval_{job}')
            
            self._starts.append(start)
            self._ends.append(end)
            self._intervals.append(interval)
        
        # Add precedence constraints
        for job in range(n_jobs):
            for succ in successors[job]:
                # End of predecessor <= Start of successor
                self.model.Add(self._ends[job] <= self._starts[succ])
        
        # Add cumulative resource constraints
        for res in range(n_resources):
            demands = []
            intervals_for_resource = []
            
            for job in range(n_jobs):
                demand = resource_requirements[job][res]
                if demand > 0:
                    demands.append(demand)
                    intervals_for_resource.append(self._intervals[job])
            
            if intervals_for_resource:
                capacity = resource_capacities[res]
                self.model.AddCumulative(intervals_for_resource, demands, capacity)
        
        # Objective: Minimize the end time of the last job (sink node)
        self.model.Minimize(self._ends[n_jobs - 1])
        
        # Solve
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        
        status = self.solver.Solve(self.model)
        
        # Process results
        status_dict = {
            cp_model.OPTIMAL: 'OPTIMAL',
            cp_model.FEASIBLE: 'FEASIBLE',
            cp_model.INFEASIBLE: 'INFEASIBLE',
            cp_model.MODEL_INVALID: 'MODEL_INVALID',
            cp_model.UNKNOWN: 'UNKNOWN'
        }
        
        status_string = status_dict.get(status, 'UNKNOWN')
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            makespan = self.solver.Value(self._ends[n_jobs - 1])
            schedule = {}
            for job in range(n_jobs):
                schedule[job + 1] = self.solver.Value(self._starts[job])
        else:
            makespan = None
            schedule = None
        
        return makespan, status_string, schedule
    
    def get_schedule_dict(self) -> Dict[int, int]:
        """
        Extract the solved schedule as a dictionary.
        
        OptiMUS Pattern: Provides solver output in format compatible with verify_schedule().
        
        Returns:
            Dictionary mapping job_id (1-indexed) to start_time.
            Empty dict if no solution available.
        """
        if self.solver is None or not self._starts:
            return {}
        
        schedule = {}
        for job in range(len(self._starts)):
            try:
                start_time = self.solver.Value(self._starts[job])
                schedule[job + 1] = start_time  # 1-indexed
            except:
                pass
        
        return schedule
    
    def get_solution_details(self, data):
        """
        Get detailed solution including start times for all jobs.
        
        Args:
            data: Parsed data dictionary
            
        Returns:
            dict: Solution details with start/end times for each job
        """
        if self.solver is None or self.model is None:
            return None
        
        n_jobs = data['n_jobs']
        solution = {}
        
        for job in range(n_jobs):
            start_var = self.model.GetIntVarFromProtoIndex(job * 3)
            end_var = self.model.GetIntVarFromProtoIndex(job * 3 + 2)
            
            solution[job] = {
                'job_id': job + 1,  # Convert back to 1-indexed
                'start': self.solver.Value(start_var) if hasattr(start_var, 'Index') else 0,
                'end': self.solver.Value(end_var) if hasattr(end_var, 'Index') else 0,
                'duration': data['durations'][job]
            }
        
        return solution

    def verify_schedule(
        self, 
        data: dict, 
        schedule_dict: dict
    ) -> dict:
        """
        Verify if a proposed schedule satisfies all RCPSP constraints.
        
        OptiMUS Pattern: Verification Layer - validates LLM-proposed schedules
        against precedence and resource capacity constraints WITHOUT solving.
        
        This method is crucial for the Feedback Loop: if infeasible, the error
        message provides specific constraint violation info for LLM refinement.
        
        Args:
            data: Parsed data dictionary from RCPSPParser containing:
                  - n_jobs: Number of jobs
                  - durations: List of job durations (0-indexed)
                  - successors: List of successor lists (0-indexed)
                  - resource_requirements: List of [R1,R2,R3,R4] per job
                  - resource_capacities: List of capacities [cap_R1, cap_R2, ...]
                  
            schedule_dict: Proposed schedule mapping job_id (1-indexed) to start_time.
                          Format: {1: 0, 2: 3, 3: 5, ...}
                          
        Returns:
            Dictionary with verification result:
            {
                'is_feasible': bool,
                'error_message': str,  # Empty if feasible, detailed if not
                'violations': List[dict],  # List of all violations found
                'makespan': int  # Calculated makespan from schedule
            }
        """
        n_jobs = data['n_jobs']
        durations = data['durations']
        successors = data['successors']
        resource_requirements = data['resource_requirements']
        resource_capacities = data['resource_capacities']
        n_resources = data['n_resources']
        
        violations = []
        
        # Convert 1-indexed schedule to 0-indexed for internal processing
        schedule_0idx = {}
        for job_id, start_time in schedule_dict.items():
            schedule_0idx[job_id - 1] = start_time
        
        # ---------------------------------------------------------------------
        # Check 1: Verify all jobs are scheduled
        # ---------------------------------------------------------------------
        missing_jobs = []
        for job in range(n_jobs):
            if job not in schedule_0idx:
                missing_jobs.append(job + 1)  # Report as 1-indexed
        
        if missing_jobs:
            violations.append({
                'type': 'MISSING_JOBS',
                'jobs': missing_jobs,
                'message': f"Missing jobs in schedule: {missing_jobs}"
            })
        
        # If too many jobs missing, return early
        if len(missing_jobs) > n_jobs // 2:
            return {
                'is_feasible': False,
                'error_message': f"Schedule incomplete: {len(missing_jobs)} jobs missing out of {n_jobs}",
                'violations': violations,
                'makespan': -1
            }
        
        # ---------------------------------------------------------------------
        # Check 2: Precedence Constraints
        # For each (predecessor, successor) pair: end[pred] <= start[succ]
        # ---------------------------------------------------------------------
        for job in range(n_jobs):
            if job not in schedule_0idx:
                continue
                
            job_start = schedule_0idx[job]
            job_duration = durations[job]
            job_end = job_start + job_duration
            
            for succ in successors[job]:
                if succ not in schedule_0idx:
                    continue
                    
                succ_start = schedule_0idx[succ]
                
                if job_end > succ_start:
                    violations.append({
                        'type': 'PRECEDENCE_VIOLATION',
                        'predecessor': job + 1,  # 1-indexed for reporting
                        'successor': succ + 1,
                        'pred_end': job_end,
                        'succ_start': succ_start,
                        'message': (
                            f"Precedence violation: Job {job + 1} ends at time {job_end}, "
                            f"but successor Job {succ + 1} starts at time {succ_start} "
                            f"(must start at or after {job_end})"
                        )
                    })
        
        # ---------------------------------------------------------------------
        # Check 3: Resource Capacity Constraints
        # At any time t: sum(demands of active jobs) <= capacity for each resource
        # ---------------------------------------------------------------------
        
        # Calculate makespan (end time of last scheduled job)
        makespan = 0
        for job in range(n_jobs):
            if job in schedule_0idx:
                job_end = schedule_0idx[job] + durations[job]
                makespan = max(makespan, job_end)
        
        # For each time point, check resource usage
        for t in range(makespan + 1):
            # Calculate resource usage at time t
            resource_usage = [0] * n_resources
            active_jobs = []
            
            for job in range(n_jobs):
                if job not in schedule_0idx:
                    continue
                    
                job_start = schedule_0idx[job]
                job_end = job_start + durations[job]
                
                # Job is active at time t if start <= t < end
                if job_start <= t < job_end:
                    active_jobs.append(job + 1)
                    for res in range(n_resources):
                        resource_usage[res] += resource_requirements[job][res]
            
            # Check capacity violations
            for res in range(n_resources):
                if resource_usage[res] > resource_capacities[res]:
                    violations.append({
                        'type': 'CAPACITY_VIOLATION',
                        'resource': f'R{res + 1}',
                        'time': t,
                        'usage': resource_usage[res],
                        'capacity': resource_capacities[res],
                        'active_jobs': active_jobs,
                        'message': (
                            f"Resource R{res + 1} overloaded at time {t}: "
                            f"usage={resource_usage[res]}, capacity={resource_capacities[res]}. "
                            f"Active jobs: {active_jobs}"
                        )
                    })
        
        # ---------------------------------------------------------------------
        # Compile result
        # ---------------------------------------------------------------------
        is_feasible = len(violations) == 0
        
        if is_feasible:
            error_message = ""
        else:
            # Create summary error message
            precedence_violations = [v for v in violations if v['type'] == 'PRECEDENCE_VIOLATION']
            capacity_violations = [v for v in violations if v['type'] == 'CAPACITY_VIOLATION']
            missing_violations = [v for v in violations if v['type'] == 'MISSING_JOBS']
            
            error_parts = []
            
            if missing_violations:
                error_parts.append(missing_violations[0]['message'])
            
            if precedence_violations:
                # Report first precedence violation in detail
                first_prec = precedence_violations[0]
                error_parts.append(first_prec['message'])
                if len(precedence_violations) > 1:
                    error_parts.append(
                        f"(+{len(precedence_violations) - 1} more precedence violations)"
                    )
            
            if capacity_violations:
                # Report first capacity violation in detail
                first_cap = capacity_violations[0]
                error_parts.append(first_cap['message'])
                if len(capacity_violations) > 1:
                    error_parts.append(
                        f"(+{len(capacity_violations) - 1} more capacity violations)"
                    )
            
            error_message = " | ".join(error_parts)
        
        return {
            'is_feasible': is_feasible,
            'error_message': error_message,
            'violations': violations,
            'makespan': makespan
        }


class RCPSPVerifier:
    """
    Verifies a given RCPSP schedule against constraints.
    Provides detailed feedback on violations.
    """
    
    def verify(self, data, schedule):
        """
        Verify the schedule.
        
        Args:
            data: Parsed data dictionary (from RCPSPParser or parse_psplib)
            schedule: Dict mapping job_id (1-indexed) to start_time
            
        Returns:
            dict: {
                'is_feasible': bool,
                'errors': list of strings describing violations
            }
        """
        # Normalize data structure
        if 'metadata' in data and 'jobs' in data and isinstance(data['jobs'], dict):
            # Data from parse_psplib
            n_jobs = data['metadata']['jobs']
            n_resources = data['metadata']['resources']
            
            # Convert jobs dict to lists for easier indexing
            job_ids = sorted(data['jobs'].keys())
            durations = [data['jobs'][jid]['duration'] for jid in job_ids]
            
            # Map resource names (R1, R2...) to indices
            res_names = sorted(data['resources'].keys())
            resource_capacities = [data['resources'][rn] for rn in res_names]
            
            resource_requirements = []
            for jid in job_ids:
                reqs = [data['jobs'][jid]['demands'].get(rn, 0) for rn in res_names]
                resource_requirements.append(reqs)
                
            successors = []
            for jid in job_ids:
                # Convert successor IDs to 0-indexed indices
                succ_ids = data['jobs'][jid]['successors']
                successors.append([job_ids.index(sid) for sid in succ_ids if sid in job_ids])
        else:
            # Data from RCPSPParser
            n_jobs = data['n_jobs']
            n_resources = data['n_resources']
            durations = data['durations']
            resource_requirements = data['resource_requirements']
            successors = data['successors']
            resource_capacities = data['resource_capacities']
        
        errors = []
        
        # 1. Check if all jobs are scheduled
        for i in range(n_jobs):
            job_id = i + 1
            if job_id not in schedule:
                errors.append(f"Job {job_id} is missing from the schedule.")
        
        if errors:
            return {'is_feasible': False, 'errors': errors}
            
        # 2. Check precedence constraints
        for i in range(n_jobs):
            job_id = i + 1
            start_i = schedule[job_id]
            if isinstance(start_i, dict):
                # Handle case where LLM might return a dict like {"start": 0}
                start_i = start_i.get('start', 0)
            
            end_i = start_i + durations[i]
            
            for succ_idx in successors[i]:
                succ_id = succ_idx + 1
                start_succ = schedule[succ_id]
                if end_i > start_succ:
                    errors.append(f"Precedence violation: Job {job_id} ends at {end_i}, but its successor Job {succ_id} starts at {start_succ}.")
        
        # 3. Check resource constraints
        # Find max end time to define the horizon for checking
        horizon = 0
        for i in range(n_jobs):
            job_id = i + 1
            horizon = max(horizon, schedule[job_id] + durations[i])
            
        for t in range(horizon + 1):
            resource_usage = [0] * n_resources
            for i in range(n_jobs):
                job_id = i + 1
                start_i = schedule[job_id]
                if isinstance(start_i, dict):
                    start_i = start_i.get('start', 0)
                
                end_i = start_i + durations[i]
                
                if start_i <= t < end_i:
                    for r in range(n_resources):
                        resource_usage[r] += resource_requirements[i][r]
            
            for r in range(n_resources):
                if resource_usage[r] > resource_capacities[r]:
                    errors.append(f"Resource violation at time {t}: Resource R{r+1} usage is {resource_usage[r]}, but capacity is {resource_capacities[r]}.")
                    if len(errors) > 10:
                        errors.append("... and more resource violations.")
                        return {'is_feasible': False, 'errors': errors}

        return {
            'is_feasible': len(errors) == 0,
            'errors': errors
        }


def main():
    """
    Main execution: Load, parse, and solve a sample RCPSP instance.
    """
    print("=" * 70)
    print("RCPSP SOLVER - Baseline Benchmark")
    print("=" * 70)
    
    # Path to the instance file
    instance_file = os.path.join("data", "raw", "rcpsp", "j30", "j301_1.sm")
    
    if not os.path.exists(instance_file):
        print(f"\n✗ Error: File not found: {instance_file}")
        return
    
    print(f"\n[1] Loading instance: {instance_file}")
    
    # Parse the instance
    parser = RCPSPParser(instance_file)
    data = parser.parse()
    
    print(f"    - Jobs: {data['n_jobs']}")
    print(f"    - Resources: {data['n_resources']}")
    print(f"    - Resource Capacities: {data['resource_capacities']}")
    
    # Solve the instance
    print(f"\n[2] Solving with CP-SAT...")
    solver = RCPSPSolver()
    makespan, status = solver.solve(data, time_limit_seconds=300)
    
    # Print results
    print(f"\n[3] Results:")
    print(f"    - Solver Status: {status}")
    if makespan is not None:
        print(f"    - Optimal Makespan: {makespan}")
    else:
        print(f"    - No solution found")
    
    print("\n" + "=" * 70)
    print("SOLVER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
