"""
Main Simulation Entry Point
ICDAM 2025 - Integration Test for Hybrid LLM Multi-Agent System

This module demonstrates the data-driven simulation workflow with LLM negotiation:
1. Load benchmark data from PSPLib files
2. Initialize state-aware agents with real project data
3. Execute LLM-driven resource negotiation between agents
4. LLM-Solver-Loop for schedule verification and repair
"""

import os
import sys
from typing import Dict, Any

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.psplib_loader import parse_psplib, create_dummy_sm_file
from agents.basic_agents import WarehouseAgent, ProjectManagerAgent


def run_simulation(data_file: str) -> Dict[str, Any]:
    """
    Execute the main simulation workflow with LLM-driven negotiation and disruption.
    """
    print("=" * 70)
    print("ICDAM 2025 - Multi-Agent System Simulation")
    print("Phase 3: Dynamic & Disruption Scenarios")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Benchmark Data (Table 2)
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading Benchmark Data")
    print("-" * 70)
    
    if not os.path.exists(data_file):
        create_dummy_sm_file(data_file)
    
    try:
        project_data = parse_psplib(data_file)
        print(f"  [OK] Loaded: {project_data['metadata']['file']}")
    except Exception as e:
        print(f"  [ERROR] Failed to parse file: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize State-Aware Agents (Table 1)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Initializing State-Aware Agents")
    print("-" * 70)
    
    resource_capacities = project_data['resources']
    job_data = project_data['jobs']
    
    warehouse = WarehouseAgent(name="Warehouse_Alpha", capacity_data=resource_capacities)
    pm = ProjectManagerAgent(name="PM_Beta", project_data=job_data, resource_capacities=resource_capacities, data_file_path=data_file)
    
    print(f"  [OK] Agents initialized: {warehouse.name}, {pm.name}")
    
    # -------------------------------------------------------------------------
    # Step 3: LLM-Driven Negotiation Layer (Table 1 & 2)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] LLM-Driven Negotiation Layer")
    print("-" * 70)
    
    # Simulate a DISRUPTION scenario (Phase 3)
    disruption_active = True
    if disruption_active:
        print("\n[DISRUPTION] Resource Failure Detected: R1 capacity reduced by 50%!")
        warehouse.inventory['R1'] //= 2
        warehouse._build_knowledge_base() 
    
    target_job_id = 2
    print(f"\n  [Context] PM requesting resources for Job #{target_job_id}")
    
    request_message = pm.request_resources(job_id=target_job_id, warehouse=warehouse)
    response_message = warehouse.process_request(request_message)
    
    print(f"\n  [Warehouse Response Type]: {response_message.get('type')}")
    print(f"  [Warehouse Response Content]: {response_message.get('content')[:100]}...")
    
    # -------------------------------------------------------------------------
    # Step 4: LLM-Solver-Loop (Verify & Repair)
    # -------------------------------------------------------------------------
    print("\n[STEP 4] LLM-Solver-Loop: Verify & Repair")
    print("-" * 70)
    
    from solvers.rcpsp_solver import RCPSPVerifier
    from utils.parser import JSONParser
    
    verifier = RCPSPVerifier()
    jobs_to_schedule = sorted(list(job_data.keys()))[:5]
    
    pm_prompt = f"""Propose a JSON schedule for jobs {jobs_to_schedule}. 
RESOURCE CONSTRAINTS (Global Capacities):
{pm.resource_capacities}

JOB DATA (Durations and Demands):
{ {jid: pm.get_job_details(jid) for jid in jobs_to_schedule} }

Instructions:
1. Respect precedence constraints (successors).
2. Respect resource constraints (total demand at any time t <= capacity).
3. Provide the schedule in JSON format.
"""
    
    max_attempts = 3
    current_attempt = 1
    is_feasible = False
    
    while current_attempt <= max_attempts and not is_feasible:
        print(f"  [Attempt {current_attempt}] PM generating schedule...")
        llm_response = pm.brain.think(pm_prompt)
        print(f"  [DEBUG] LLM Response: {llm_response[:200]}...")
        parsed_schedule = JSONParser.parse_llm_response(llm_response)
        print(f"  [DEBUG] Parsed Schedule: {parsed_schedule}")
        
        proposed_schedule = {}
        
        # Helper to extract start time from various value formats
        def get_start_time(val):
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, dict) and "start_time" in val:
                return int(val["start_time"])
            return None

        # 1. Check top-level keys and common nested keys
        search_dicts = [parsed_schedule]
        if "function" in parsed_schedule and isinstance(parsed_schedule["function"], dict):
            search_dicts.append(parsed_schedule["function"])
        if "schedule" in parsed_schedule and isinstance(parsed_schedule["schedule"], dict):
            search_dicts.append(parsed_schedule["schedule"])
            
        for d in search_dicts:
            for k, v in d.items():
                job_id = None
                if k.isdigit():
                    job_id = int(k)
                elif k.lower().startswith("job_") and k[4:].isdigit():
                    job_id = int(k[4:])
                elif k.lower().startswith("job ") and k[4:].isdigit():
                    job_id = int(k[4:])
                
                if job_id is not None:
                    st = get_start_time(v)
                    if st is not None:
                        proposed_schedule[job_id] = st

        # 2. Handle list-based formats (e.g., in "function" or "schedule" or top-level if it was a list)
        search_lists = []
        if isinstance(parsed_schedule.get("function"), list):
            search_lists.append(parsed_schedule["function"])
        if isinstance(parsed_schedule.get("schedule"), list):
            search_lists.append(parsed_schedule["schedule"])
        # JSONParser puts the list in "function" if the whole block was a list
            
        for l in search_lists:
            for item in l:
                if isinstance(item, dict):
                    jid = item.get("job_id") or item.get("id")
                    st = item.get("start_time") or item.get("start")
                    if jid is not None and st is not None:
                        try:
                            proposed_schedule[int(jid)] = int(st)
                        except (ValueError, TypeError):
                            continue
            
        if proposed_schedule:
            full_schedule = dict(pm.optimal_schedule) if pm.optimal_schedule else {jid: 0 for jid in job_data.keys()}
            full_schedule.update(proposed_schedule)
            
            verification_result = verifier.verify(project_data, full_schedule)
            if verification_result['is_feasible']:
                print("  [RESULT] Schedule is FEASIBLE!")
                is_feasible = True
            else:
                print(f"  [RESULT] Schedule is INFEASIBLE: {verification_result['errors'][0]}")
                pm_prompt = f"Repair this schedule. Errors: {verification_result['errors'][0]}"
                current_attempt += 1
        else:
            print("  [ERROR] Could not parse schedule.")
            current_attempt += 1

    # -------------------------------------------------------------------------
    # Step 6: Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    return {
        'negotiation': response_message.get('type'),
        'feasibility': is_feasible,
        'attempts': current_attempt if is_feasible else max_attempts,
        'disruption': disruption_active
    }


def main() -> None:
    data_file = os.path.join("data", "raw", "rcpsp", "j30", "j3010_1.sm")
    run_simulation(data_file)


if __name__ == "__main__":
    main()
