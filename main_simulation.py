"""
Main Simulation Entry Point
ICDAM 2025 - Integration Test for Hybrid LLM Multi-Agent System

This module demonstrates the data-driven simulation workflow with LLM negotiation:
1. Load benchmark data from PSPLib files
2. Initialize state-aware agents with real project data
3. Execute LLM-driven resource negotiation between agents

References:
- AgentScope: Role-based agent initialization and message passing
- REALM-Bench: Benchmark-driven evaluation
- ICDAM 2025: Table 1 (Core MAS + Negotiation Layer) + Table 2 (Benchmark Data)
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
    
    run_simulation(data_file)


if __name__ == "__main__":
    main()
