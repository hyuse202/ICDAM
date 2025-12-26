"""
RCPSP Solver using Google OR-Tools CP-SAT
Solves Resource-Constrained Project Scheduling Problem from PSPLib instances.
"""

import os
import re
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
        
        # Decision variables
        starts = []
        ends = []
        intervals = []
        
        # Create interval variables for each job
        for job in range(n_jobs):
            start = self.model.NewIntVar(0, horizon, f'start_{job}')
            duration = durations[job]
            end = self.model.NewIntVar(0, horizon, f'end_{job}')
            interval = self.model.NewIntervalVar(start, duration, end, f'interval_{job}')
            
            starts.append(start)
            ends.append(end)
            intervals.append(interval)
        
        # Add precedence constraints
        for job in range(n_jobs):
            for succ in successors[job]:
                # End of predecessor <= Start of successor
                self.model.Add(ends[job] <= starts[succ])
        
        # Add cumulative resource constraints
        for res in range(n_resources):
            demands = []
            intervals_for_resource = []
            
            for job in range(n_jobs):
                demand = resource_requirements[job][res]
                if demand > 0:
                    demands.append(demand)
                    intervals_for_resource.append(intervals[job])
            
            if intervals_for_resource:
                capacity = resource_capacities[res]
                self.model.AddCumulative(intervals_for_resource, demands, capacity)
        
        # Objective: Minimize the end time of the last job (sink node)
        self.model.Minimize(ends[n_jobs - 1])
        
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
            makespan = self.solver.Value(ends[n_jobs - 1])
            schedule = {}
            for job in range(n_jobs):
                schedule[job + 1] = self.solver.Value(starts[job])
        else:
            makespan = None
            schedule = None
        
        return makespan, status_string, schedule
    
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
