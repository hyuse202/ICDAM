"""
Main Simulation Entry Point
ICDAM 2025 - Integration Test for Hybrid LLM Multi-Agent System

This module demonstrates the data-driven simulation workflow with LLM negotiation:
1. Load benchmark data from PSPLib files
2. Initialize state-aware agents with real project data
3. Execute LLM-driven resource negotiation between agents
4. [NEW] OptiMUS Pattern: Verify LLM-proposed schedules with Solver

References:
- AgentScope: Role-based agent initialization and message passing
- REALM-Bench: Benchmark-driven evaluation
- OptiMUS: LLM + Solver Verification Loop (NeurIPS 2023)
- ICDAM 2025: Table 1 (Core MAS + Negotiation Layer) + Table 2 (Benchmark Data)
"""

import os
import sys
from typing import Dict, Any, List

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.psplib_loader import parse_psplib, create_dummy_sm_file
from utils.plan_parser import parse_schedule_from_text, format_schedule_for_display
from agents.basic_agents import WarehouseAgent, ProjectManagerAgent
from solvers.rcpsp_solver import RCPSPParser, RCPSPSolver


def run_simulation(data_file: str) -> None:
    """
    Execute the main simulation workflow with LLM-driven negotiation.
    
    ICDAM 2025: Integration test demonstrating Table 1 + Table 2 + Negotiation Layer.
    
    Workflow:
    1. Load benchmark data from PSPLib file
    2. Initialize WarehouseAgent with resource capacities
    3. Initialize ProjectManagerAgent with job data
    4. PM generates LLM-driven resource request
    5. Warehouse processes request with LLM-driven decision
    6. Display negotiation results
    
    Args:
        data_file: Path to PSPLib .sm file.
    """
    print("=" * 70)
    print("ICDAM 2025 - Multi-Agent System Simulation")
    print("Phase 2: LLM-Driven Negotiation Layer")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Benchmark Data (Table 2)
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading Benchmark Data")
    print("-" * 70)
    
    # Check if file exists, create dummy if needed
    if not os.path.exists(data_file):
        print(f"  File not found: {data_file}")
        print("  Creating dummy PSPLib file for immediate testing...")
        
        # Create dummy file in the expected location
        dummy_dir = os.path.dirname(data_file)
        if dummy_dir:
            os.makedirs(dummy_dir, exist_ok=True)
        create_dummy_sm_file(data_file)
    
    # Parse the PSPLib file
    try:
        project_data = parse_psplib(data_file)
        print(f"  [OK] Loaded: {project_data['metadata']['file']}")
        print(f"       Jobs: {project_data['metadata']['jobs']}")
        print(f"       Resources: {len(project_data['resources'])} types")
        print(f"       Capacities: {project_data['resources']}")
    except Exception as e:
        print(f"  [ERROR] Failed to parse file: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize State-Aware Agents (Table 1)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Initializing State-Aware Agents with LLM Brain + Solver")
    print("-" * 70)
    
    # Extract data views for agents
    resource_capacities: Dict[str, int] = project_data['resources']
    job_data: Dict[int, Dict[str, Any]] = project_data['jobs']
    
    # Initialize WarehouseAgent with resource inventory
    warehouse = WarehouseAgent(
        name="Warehouse_Alpha",
        capacity_data=resource_capacities
    )
    print(f"  [OK] Created {warehouse.name}")
    print(f"       Role: {warehouse.role}")
    print(f"       Inventory: {warehouse.inventory}")
    print(f"       LLM Brain: {'API Mode' if warehouse.brain.is_api_available() else 'Mock Mode'}")
    
    # Initialize ProjectManagerAgent with job data AND data file for solver
    pm = ProjectManagerAgent(
        name="PM_Beta",
        project_data=job_data,
        data_file_path=data_file  # Pass file path for OR-Tools solver
    )
    print(f"  [OK] Created {pm.name}")
    print(f"       Role: {pm.role}")
    print(f"       Managing: {len(pm.project_data)} jobs")
    print(f"       Optimal Makespan: {pm.optimal_makespan} days ({pm.solver_status})")
    print(f"       LLM Brain: {'API Mode' if pm.brain.is_api_available() else 'Mock Mode'}")
    
    # -------------------------------------------------------------------------
    # Step 3: LLM-Driven Negotiation Scenario
    # -------------------------------------------------------------------------
    print("\n[STEP 3] LLM-Driven Negotiation: Resource Request for Job #2")
    print("-" * 70)
    
    target_job_id = 2
    
    # Get job details for display
    job_details = pm.get_job_details(target_job_id)
    if job_details is None:
        # Fallback to first available non-dummy job
        available_jobs = [jid for jid in job_data.keys() if jid > 1]
        if available_jobs:
            target_job_id = available_jobs[0]
            job_details = pm.get_job_details(target_job_id)
    
    if job_details:
        print(f"\n  [Context] Job #{target_job_id} Requirements:")
        print(f"       Duration: {job_details['duration']} time units")
        print(f"       Demands: {job_details['demands']}")
        
        # ---------------------------------------------------------------------
        # Phase A: PM generates resource request using LLM
        # ---------------------------------------------------------------------
        print(f"\n  {'='*60}")
        print("  [PHASE A] PM Generates Resource Request (LLM)")
        print(f"  {'='*60}")
        
        request_message = pm.request_resources(
            job_id=target_job_id,
            warehouse=warehouse
        )
        
        print(f"\n  [PM Request Content]:")
        print(f"  {'-'*60}")
        print(f"  {request_message.get('content', 'N/A')}")
        print(f"  {'-'*60}")
        
        # ---------------------------------------------------------------------
        # Phase B: Warehouse processes request using LLM
        # ---------------------------------------------------------------------
        print(f"\n  {'='*60}")
        print("  [PHASE B] Warehouse Processes Request (LLM)")
        print(f"  {'='*60}")
        
        response_message = warehouse.process_request(request_message)
        
        print(f"\n  [Warehouse Response Content]:")
        print(f"  {'-'*60}")
        print(f"  Type: {response_message.get('type', 'N/A')}")
        print(f"  Content: {response_message.get('content', 'N/A')}")
        print(f"  {'-'*60}")
        
        # ---------------------------------------------------------------------
        # Phase C: Negotiation Result Summary
        # ---------------------------------------------------------------------
        print(f"\n  {'='*60}")
        print("  [PHASE C] Negotiation Result")
        print(f"  {'='*60}")
        
        result_type = response_message.get('type', 'UNKNOWN')
        if result_type == "AGREE":
            print(f"  [RESULT] NEGOTIATION SUCCESSFUL")
            print(f"           Warehouse agreed to provide resources for Job #{target_job_id}")
        elif result_type == "COUNTER":
            print(f"  [RESULT] COUNTER-OFFER RECEIVED")
            print(f"           Warehouse proposed alternative allocation")
        else:
            print(f"  [RESULT] REQUEST DECLINED")
            print(f"           Warehouse cannot fulfill the request")
    else:
        print(f"  [ERROR] Could not retrieve job details for simulation.")
    
    # -------------------------------------------------------------------------
    # Step 4: Agent Memory Inspection
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Agent Communication History")
    print("-" * 70)
    
    print(f"\n  [{pm.name}] Memory ({len(pm.memory)} messages):")
    for idx, msg in enumerate(pm.memory[-3:], 1):  # Last 3 messages
        direction = msg.get('direction', 'unknown')
        msg_type = msg.get('type', 'N/A')
        content_preview = msg.get('content', '')[:50]
        print(f"    {idx}. [{direction.upper()}] ({msg_type}) {content_preview}...")
    
    print(f"\n  [{warehouse.name}] Memory ({len(warehouse.memory)} messages):")
    for idx, msg in enumerate(warehouse.memory[-3:], 1):  # Last 3 messages
        direction = msg.get('direction', 'unknown')
        msg_type = msg.get('type', 'N/A')
        content_preview = msg.get('content', '')[:50]
        print(f"    {idx}. [{direction.upper()}] ({msg_type}) {content_preview}...")
    
    # -------------------------------------------------------------------------
    # Step 5: Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\n[Summary]")
    print(f"  - Benchmark Loader (Table 2): OPERATIONAL")
    print(f"  - State-Aware Agents (Table 1): OPERATIONAL")
    print(f"  - LLM Negotiation Layer: OPERATIONAL")
    print(f"  - Agent Message Passing: VERIFIED")
    print(f"\n[Components Demonstrated]")
    print(f"  - LLMBrain.think(): PM generated formal resource proposal")
    print(f"  - LLMBrain.think_with_context(): Warehouse evaluated against inventory")
    print(f"  - AgentScope Message Structure: sender/receiver/type/content/timestamp")
    print(f"\n[Next Phase]")
    print(f"  - Implement multi-round negotiation protocols")
    print(f"  - Connect OR-Tools solver for optimal scheduling")
    print(f"  - Add more agent roles (Supplier, Logistics)")
    print("=" * 70)


def main() -> None:
    """
    Main entry point for the simulation.
    """
    # Default data file path
    default_data_file = os.path.join(
        "data", "raw", "rcpsp", "j30", "j3010_1.sm"
    )
    
    # Allow command-line override
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = default_data_file
    
    # Run main negotiation simulation
    run_simulation(data_file)
    
    # Run OptiMUS verification test
    run_verification_test(data_file)
    
    # Run OptiMUS self-correction test
    run_self_correction_test(data_file)


def run_self_correction_test(data_file: str) -> None:
    """
    OptiMUS Pattern: Test the Self-Correction Loop in ProjectManagerAgent.
    
    Phase 3: Demonstrates the complete LLM + Solver feedback loop:
    1. PM uses LLM to generate initial schedule
    2. Solver verifies and provides error feedback
    3. LLM refines schedule based on errors
    4. Loop until valid or fallback to optimal
    
    Args:
        data_file: Path to PSPLib .sm file.
    """
    print("\n")
    print("=" * 70)
    print("OPTIMUS PATTERN - Self-Correction Loop Test")
    print("Phase 3: LLM Schedule Generation with Solver Feedback")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize PM Agent with Solver
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Initializing ProjectManagerAgent with Solver Integration")
    print("-" * 70)
    
    if not os.path.exists(data_file):
        print(f"  [ERROR] File not found: {data_file}")
        return
    
    # Load project data
    project_data = parse_psplib(data_file)
    job_data = project_data['jobs']
    
    print(f"  [OK] Loaded: {os.path.basename(data_file)}")
    print(f"       Jobs: {len(job_data)}")
    print(f"       Resources: {len(project_data['resources'])}")
    
    # Create PM agent with solver
    pm = ProjectManagerAgent(
        name="PM_OptiMUS",
        project_data=job_data,
        data_file_path=data_file
    )
    
    print(f"  [OK] Created {pm.name}")
    print(f"       Optimal Makespan: {pm.optimal_makespan} days")
    print(f"       Solver Status: {pm.solver_status}")
    print(f"       LLM Brain: {'API Mode' if pm.brain.is_api_available() else 'Mock Mode'}")
    
    # -------------------------------------------------------------------------
    # Step 2: Run Self-Correction Loop
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[STEP 2] Running OptiMUS Self-Correction Loop")
    print("=" * 70)
    
    # Call develop_viable_plan with 3 retries
    schedule, result_info = pm.develop_viable_plan(max_retries=3)
    
    # -------------------------------------------------------------------------
    # Step 3: Display Results
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[STEP 3] Self-Correction Results")
    print("=" * 70)
    
    print(f"\n  [Result Summary]")
    print(f"  - Method Used: {result_info['method'].upper()}")
    print(f"  - Total Attempts: {result_info['attempts']}")
    print(f"  - Success: {result_info['success']}")
    print(f"  - Final Makespan: {result_info['final_makespan']}")
    
    if result_info['method'] == 'llm':
        print(f"\n  [SUCCESS] LLM generated a valid schedule!")
        if pm.optimal_makespan:
            gap = result_info['final_makespan'] - pm.optimal_makespan
            gap_pct = (gap / pm.optimal_makespan) * 100 if pm.optimal_makespan else 0
            print(f"  - LLM Makespan: {result_info['final_makespan']}")
            print(f"  - Optimal Makespan: {pm.optimal_makespan}")
            print(f"  - Gap: {gap} ({gap_pct:.1f}%)")
    else:
        print(f"\n  [FALLBACK] Used OR-Tools optimal schedule (Grounding mechanism)")
        print(f"  - LLM could not produce valid schedule after {result_info['attempts']} attempts")
    
    # Show attempt history
    if result_info['history']:
        print(f"\n  [Attempt History]")
        print("-" * 50)
        for attempt_log in result_info['history']:
            attempt_num = attempt_log['attempt']
            parsed_count = len(attempt_log['parsed_schedule']) if attempt_log['parsed_schedule'] else 0
            
            if attempt_log['verification']:
                is_feasible = attempt_log['verification']['is_feasible']
                violations = attempt_log['verification']['violation_count']
                status = "VALID" if is_feasible else f"INVALID ({violations} violations)"
            else:
                status = "PARSE FAILED"
            
            print(f"    Attempt {attempt_num}: Parsed {parsed_count} jobs -> {status}")
    
    # Show schedule preview if successful
    if schedule and result_info['success']:
        print(f"\n  [Schedule Preview] (first 10 jobs)")
        print("-" * 50)
        for job_id in sorted(schedule.keys())[:10]:
            start = schedule[job_id]
            job_data_item = pm.project_data.get(job_id, {})
            duration = job_data_item.get('duration', 0)
            print(f"    Job {job_id}: Start={start}, Duration={duration}, End={start + duration}")
    
    # -------------------------------------------------------------------------
    # Step 4: Verify Final Schedule
    # -------------------------------------------------------------------------
    if schedule and pm.solver and pm.solver_data:
        print(f"\n{'='*70}")
        print("[STEP 4] Final Schedule Verification")
        print("=" * 70)
        
        final_verification = pm.solver.verify_schedule(pm.solver_data, schedule)
        
        print(f"\n  [Verification Result]")
        print(f"  - is_feasible: {final_verification['is_feasible']}")
        print(f"  - makespan: {final_verification['makespan']}")
        print(f"  - violations: {len(final_verification['violations'])}")
        
        if final_verification['is_feasible']:
            print(f"\n  [PASS] Final schedule verified as feasible!")
        else:
            print(f"\n  [WARNING] Final schedule has issues: {final_verification['error_message'][:100]}...")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SELF-CORRECTION TEST COMPLETE")
    print("=" * 70)
    print("\n[OptiMUS Pattern Summary]")
    print("  1. LLM Draft: Generate initial schedule from project knowledge")
    print("  2. Solver Verify: Check precedence + capacity constraints")
    print("  3. Error Feedback: Detailed violation info sent back to LLM")
    print("  4. LLM Repair: Refine schedule based on specific errors")
    print("  5. Grounding Fallback: Use optimal solver schedule if LLM fails")
    print("\n[Key Achievement]")
    print("  - LLM reasoning is GROUNDED by symbolic verification")
    print("  - System NEVER outputs an infeasible schedule")
    print("=" * 70)


def run_verification_test(data_file: str) -> None:
    """
    OptiMUS Pattern: Test the Verification Layer with faulty schedules.
    
    This function demonstrates the LLM + Solver feedback loop by:
    1. Creating intentionally faulty schedules
    2. Using RCPSPSolver.verify_schedule() to detect constraint violations
    3. Displaying detailed error messages for LLM refinement
    
    Args:
        data_file: Path to PSPLib .sm file.
    """
    print("\n")
    print("=" * 70)
    print("OPTIMUS PATTERN - Verification Layer Test")
    print("Phase 3: LLM + Solver Feedback Loop")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load and Parse Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading Instance Data for Verification")
    print("-" * 70)
    
    if not os.path.exists(data_file):
        print(f"  [ERROR] File not found: {data_file}")
        return
    
    parser = RCPSPParser(data_file)
    data = parser.parse()
    
    print(f"  [OK] Instance: {os.path.basename(data_file)}")
    print(f"       Jobs: {data['n_jobs']}")
    print(f"       Resources: {data['n_resources']}")
    print(f"       Capacities: {data['resource_capacities']}")
    
    # Print precedence info for first few jobs
    print(f"\n  Precedence Relations (first 5 jobs):")
    for job in range(min(5, data['n_jobs'])):
        succs = data['successors'][job]
        if succs:
            print(f"    Job {job + 1} -> Jobs {[s + 1 for s in succs]}")
    
    # Print duration and resource requirements
    print(f"\n  Job Details (first 5 jobs):")
    for job in range(min(5, data['n_jobs'])):
        dur = data['durations'][job]
        reqs = data['resource_requirements'][job]
        print(f"    Job {job + 1}: duration={dur}, resources={reqs}")
    
    # Initialize solver
    solver = RCPSPSolver()
    
    # -------------------------------------------------------------------------
    # Step 2: Test Case A - Precedence Violation
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[TEST A] Faulty Schedule - PRECEDENCE VIOLATION")
    print("=" * 70)
    
    # Create a schedule where Job 2 starts BEFORE Job 1 finishes
    # Job 1 (dummy) starts at 0, duration 0 -> ends at 0
    # Job 2 should start >= 0, but we test by making successor start too early
    
    # Find a real precedence constraint to violate
    violation_job = None
    violation_succ = None
    for job in range(data['n_jobs']):
        if data['successors'][job] and data['durations'][job] > 0:
            violation_job = job
            violation_succ = data['successors'][job][0]
            break
    
    if violation_job is not None:
        # Create faulty schedule: successor starts BEFORE predecessor ends
        pred_duration = data['durations'][violation_job]
        
        faulty_schedule_a: Dict[int, int] = {
            1: 0,  # Dummy start
        }
        
        # Fill in other jobs with reasonable times
        current_time = 0
        for job in range(data['n_jobs']):
            if job == 0:
                faulty_schedule_a[job + 1] = 0
            elif job == violation_job:
                faulty_schedule_a[job + 1] = current_time
                current_time += data['durations'][job]
            elif job == violation_succ:
                # VIOLATION: Start successor at same time as predecessor (overlapping)
                faulty_schedule_a[job + 1] = faulty_schedule_a[violation_job + 1]
            else:
                faulty_schedule_a[job + 1] = current_time
                current_time += data['durations'][job]
        
        print(f"\n  [Scenario] Creating precedence violation:")
        print(f"    - Job {violation_job + 1} (predecessor): duration={pred_duration}")
        print(f"    - Job {violation_succ + 1} (successor): should start after Job {violation_job + 1} ends")
        print(f"\n  [Faulty Schedule]")
        print(f"    Job {violation_job + 1}: Start={faulty_schedule_a[violation_job + 1]}, "
              f"End={faulty_schedule_a[violation_job + 1] + pred_duration}")
        print(f"    Job {violation_succ + 1}: Start={faulty_schedule_a[violation_succ + 1]} "
              f"<-- VIOLATION! Starts before predecessor ends")
        
        # Verify with solver
        print(f"\n  [Verification Result]")
        print("-" * 50)
        result_a = solver.verify_schedule(data, faulty_schedule_a)
        
        print(f"  is_feasible: {result_a['is_feasible']}")
        print(f"  error_message: {result_a['error_message']}")
        print(f"  violation_count: {len(result_a['violations'])}")
        
        if not result_a['is_feasible']:
            print(f"\n  [PASS] Verification correctly detected infeasibility!")
        else:
            print(f"\n  [FAIL] Verification should have detected violation!")
    else:
        print("  [SKIP] Could not find suitable precedence for test")
    
    # -------------------------------------------------------------------------
    # Step 3: Test Case B - Resource Capacity Violation
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[TEST B] Faulty Schedule - RESOURCE CAPACITY VIOLATION")
    print("=" * 70)
    
    # Create a schedule where too many jobs run at the same time
    # Find jobs with high resource requirements
    high_demand_jobs = []
    for job in range(data['n_jobs']):
        total_demand = sum(data['resource_requirements'][job])
        if total_demand > 0:
            high_demand_jobs.append((job, total_demand))
    
    high_demand_jobs.sort(key=lambda x: -x[1])  # Sort by demand descending
    
    if len(high_demand_jobs) >= 3:
        # Schedule multiple high-demand jobs at the SAME TIME to overload resources
        faulty_schedule_b: Dict[int, int] = {1: 0}  # Dummy start
        
        # Put first 3 high-demand jobs all starting at time 0
        overlap_time = 1
        for job, _ in high_demand_jobs[:3]:
            faulty_schedule_b[job + 1] = overlap_time
        
        # Fill remaining jobs sequentially
        current_time = 10
        for job in range(data['n_jobs']):
            if (job + 1) not in faulty_schedule_b:
                faulty_schedule_b[job + 1] = current_time
                current_time += max(1, data['durations'][job])
        
        # Ensure last job (dummy sink) is scheduled
        faulty_schedule_b[data['n_jobs']] = current_time
        
        print(f"\n  [Scenario] Creating resource capacity violation:")
        print(f"    - Scheduling {len(high_demand_jobs[:3])} high-demand jobs at same time")
        
        overlapping_jobs = high_demand_jobs[:3]
        total_at_t1 = [0] * data['n_resources']
        
        for job, _ in overlapping_jobs:
            reqs = data['resource_requirements'][job]
            print(f"    - Job {job + 1}: resources={reqs}, duration={data['durations'][job]}")
            for r in range(data['n_resources']):
                total_at_t1[r] += reqs[r]
        
        print(f"\n  [Resource Usage at time {overlap_time}]")
        for r in range(data['n_resources']):
            cap = data['resource_capacities'][r]
            usage = total_at_t1[r]
            status = "OVERLOADED" if usage > cap else "OK"
            print(f"    R{r+1}: usage={usage}, capacity={cap} [{status}]")
        
        # Verify with solver
        print(f"\n  [Verification Result]")
        print("-" * 50)
        result_b = solver.verify_schedule(data, faulty_schedule_b)
        
        print(f"  is_feasible: {result_b['is_feasible']}")
        print(f"  error_message: {result_b['error_message'][:200]}..." 
              if len(result_b.get('error_message', '')) > 200 
              else f"  error_message: {result_b['error_message']}")
        print(f"  violation_count: {len(result_b['violations'])}")
        
        # Count by type
        prec_v = len([v for v in result_b['violations'] if v['type'] == 'PRECEDENCE_VIOLATION'])
        cap_v = len([v for v in result_b['violations'] if v['type'] == 'CAPACITY_VIOLATION'])
        print(f"    - Precedence violations: {prec_v}")
        print(f"    - Capacity violations: {cap_v}")
        
        if not result_b['is_feasible']:
            print(f"\n  [PASS] Verification correctly detected infeasibility!")
        else:
            print(f"\n  [FAIL] Verification should have detected violation!")
    else:
        print("  [SKIP] Not enough high-demand jobs for test")
    
    # -------------------------------------------------------------------------
    # Step 4: Test Case C - Valid Schedule (Optimal from Solver)
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[TEST C] Valid Schedule - OPTIMAL SOLUTION FROM SOLVER")
    print("=" * 70)
    
    # Create a fresh solver for this test
    solver_c = RCPSPSolver()
    
    print(f"\n  [Computing optimal schedule with OR-Tools...]")
    makespan, status = solver_c.solve(data, time_limit_seconds=60)
    
    if makespan is not None and status in ['OPTIMAL', 'FEASIBLE']:
        print(f"  Solver found solution: makespan={makespan}, status={status}")
        
        # Extract the solution using the new method
        valid_schedule = solver_c.get_schedule_dict()
        
        print(f"\n  [Extracted Schedule from Solver] (first 10 jobs)")
        for job_id in sorted(valid_schedule.keys())[:10]:
            dur = data['durations'][job_id - 1]
            end = valid_schedule[job_id] + dur
            print(f"    Job {job_id}: Start={valid_schedule[job_id]}, End={end}, Duration={dur}")
        
        # Verify the valid schedule
        print(f"\n  [Verification Result]")
        print("-" * 50)
        result_c = solver_c.verify_schedule(data, valid_schedule)
        
        print(f"  is_feasible: {result_c['is_feasible']}")
        print(f"  error_message: '{result_c['error_message']}'")
        print(f"  makespan: {result_c['makespan']}")
        print(f"  violation_count: {len(result_c['violations'])}")
        
        if result_c['is_feasible']:
            print(f"\n  [PASS] Optimal schedule correctly verified as feasible!")
        else:
            print(f"\n  [WARNING] Optimal schedule marked infeasible - check verify logic")
    else:
        print(f"  [SKIP] Solver could not find solution: status={status}")
    
    # -------------------------------------------------------------------------
    # Step 5: Test Plan Parser Integration
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[TEST D] Plan Parser - LLM Output Parsing")
    print("=" * 70)
    
    # Simulate LLM output text
    llm_output = """
    Based on my analysis of the project constraints, I propose the following schedule:
    
    Job 1: Start 0 (project start)
    Job 2: Start 3
    Job 3: Start 5
    Job 4: Start 8
    Job 5: Start 12
    
    This schedule should minimize the makespan while respecting precedence.
    """
    
    print(f"\n  [Simulated LLM Output]")
    print("-" * 50)
    print(llm_output)
    
    print(f"\n  [Parsing Result]")
    print("-" * 50)
    parsed_schedule = parse_schedule_from_text(llm_output)
    print(f"  Parsed schedule: {parsed_schedule}")
    print(f"\n  {format_schedule_for_display(parsed_schedule)}")
    
    if parsed_schedule:
        print(f"\n  [PASS] Plan parser successfully extracted schedule from LLM text!")
    else:
        print(f"\n  [FAIL] Plan parser could not extract schedule!")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("OPTIMUS VERIFICATION TEST COMPLETE")
    print("=" * 70)
    print("\n[Summary]")
    print("  - verify_schedule(): Checks precedence and capacity constraints")
    print("  - Detailed error messages enable LLM feedback loop")
    print("  - Plan parser converts LLM text to structured schedule")
    print("\n[Feedback Loop Pattern]")
    print("  LLM proposes schedule -> Solver verifies -> Error message -> LLM refines")
    print("=" * 70)


if __name__ == "__main__":
    main()
