"""
Basic Agent Framework for Multi-Agent System
ICDAM 2025 - Table 1: Core MAS Infrastructure + Negotiation Layer

Defines base agent structure and specialized agents for RCPSP problem solving.
Implements state-aware agents following AgentScope patterns with LLM-driven negotiation.

References:
- AgentScope: AgentBase class and Message Passing architecture
- REALM-Bench: Benchmark-driven agent evaluation
- OptiMUS: Self-Correction Loop (LLM + Solver Feedback)
- ICDAM 2025: Hybrid LLM Multi-Agent System for SCM
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Import LLMBrain for negotiation layer
from agents.llm_brain import LLMBrain

# Import Solver components for optimal scheduling
from solvers.rcpsp_solver import RCPSPSolver, RCPSPParser
from utils.parser import JSONParser

# Import Plan Parser for OptiMUS Self-Correction Loop
from utils.plan_parser import parse_schedule_from_text, format_schedule_for_display

# Conditional import for backward compatibility
try:
    from scenarios.rcpsp_scenario import RCPSPScenario
except ImportError:
    RCPSPScenario = None  # Allow module to work without scenario dependency


class BaseAgent:
    """
    Base class for all agents in the Multi-Agent System.
    Provides core functionality for message handling, reasoning, and LLM-driven negotiation.
    
    ICDAM 2025: Integrates LLMBrain for natural language negotiation capabilities.
    """
    
    def __init__(self, name, role):
        """
        Initialize the base agent with LLM Brain.
        
        Args:
            name: Unique identifier for the agent
            role: Role description (e.g., "Resource Manager", "Project Manager")
        """
        self.name = name
        self.role = role
        self.memory = []  # List of message         from solvers.rcpsp_solver import RCPSPSolver, RCPSPParserdictionaries
        self.knowledge_base = ""  # Context/system prompt
        
        # ICDAM 2025: LLM Brain for negotiation layer
        self.brain = LLMBrain()
        
    def receive_message(self, sender, content):
        """
        Receive and log a message from another agent or system.
        
        Args:
            sender: Name of the message sender
            content: Message content (string)
        """
        message = {
            'timestamp': datetime.now().isoformat(),
            'sender': sender,
            'content': content,
            'type': 'received'
        }
        self.memory.append(message)
        print(f"[{self.name}] Received message from {sender}: \"{content}\"")
    
    def send_message(self, recipient, content):
        """
        Send a message and log it in memory.
        
        Args:
            recipient: Name of the recipient
            content: Message content
        """
        message = {
            'timestamp': datetime.now().isoformat(),
            'recipient': recipient,
            'content': content,
            'type': 'sent'
        }
        self.memory.append(message)
        print(f"[{self.name}] Sent message to {recipient}: \"{content}\"")
    
    def think(self):
        """
        Agent reasoning/decision-making method.
        Currently uses mock logic - will be replaced with LLM API calls.
        
        Returns:
            str: Agent's reasoning output
        """
        # Get the last received message
        received_messages = [m for m in self.memory if m['type'] == 'received']
        
        if received_messages:
            last_message = received_messages[-1]['content']
            response = f"[{self.name}] is thinking about... \"{last_message}\""
        else:
            response = f"[{self.name}] is thinking... (no messages received yet)"
        
        return response
    
    def get_context(self):
        """
        Get the agent's current context (knowledge base + recent memory).
        
        Returns:
            dict: Context information
        """
        return {
            'name': self.name,
            'role': self.role,
            'knowledge_base': self.knowledge_base,
            'memory_size': len(self.memory),
            'recent_messages': self.memory[-5:] if self.memory else []
        }
    
    def clear_memory(self):
        """Clear the agent's message memory."""
        self.memory = []
        print(f"[{self.name}] Memory cleared.")
    
    def send_negotiation(
        self, 
        receiver: 'BaseAgent', 
        content: str, 
        msg_type: str = "PROPOSE"
    ) -> Dict[str, Any]:
        """
        Send a negotiation message following AgentScope Message Structure.
        
        ICDAM 2025: Implements structured message passing for agent negotiation.
        
        Args:
            receiver: Target agent to receive the message.
            content: Message content (natural language).
            msg_type: Message type - one of "PROPOSE", "AGREE", "COUNTER", "REJECT".
            
        Returns:
            Dictionary containing the structured message.
        """
        message = {
            "sender": self.name,
            "receiver": receiver.name,
            "content": content,
            "type": msg_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log in sender's memory
        self.memory.append({
            **message,
            "direction": "sent"
        })
        
        # Deliver to receiver
        receiver.memory.append({
            **message,
            "direction": "received"
        })
        
        print(f"[{self.name}] -> [{receiver.name}] ({msg_type}): {content[:80]}...")
        
        return message


# =============================================================================
# ICDAM 2025 Table 1: State-Aware Agents
# =============================================================================

class WarehouseAgent(BaseAgent):
    """
    State-aware Warehouse Agent for resource management.
    
    ICDAM 2025 Table 1: Core MAS - WarehouseAgent Implementation.
    
    This agent maintains an inventory state loaded from benchmark data
    and provides resource availability checking functionality.
    
    Attributes:
        inventory: Dictionary mapping resource IDs (R1-R4) to available quantities.
        state: General state dictionary for extensibility.
    """
    
    def __init__(self, name: str, capacity_data: Dict[str, int]):
        """
        Initialize WarehouseAgent with resource capacity data.
        
        Args:
            name: Unique identifier for this agent.
            capacity_data: Dictionary of resource capacities from PSPLib loader.
                          Format: {'R1': 24, 'R2': 23, 'R3': 25, 'R4': 33}
        """
        super().__init__(name, role="Warehouse Manager")
        
        # ICDAM Table 1: State-aware agent with inventory
        self.inventory: Dict[str, int] = dict(capacity_data)
        self.state: Dict[str, Any] = {
            'initialized': True,
            'total_capacity': sum(capacity_data.values()),
            'resource_count': len(capacity_data)
        }
        
        # Build knowledge base
        self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> None:
        """Construct agent's knowledge base from inventory data."""
        kb_parts = [
            f"ROLE: {self.role}",
            f"NAME: {self.name}",
            "",
            "INVENTORY STATUS:",
        ]
        
        for resource, capacity in self.inventory.items():
            kb_parts.append(f"  - {resource}: {capacity} units available")
        
        kb_parts.extend([
            "",
            "RESPONSIBILITIES:",
            "  - Track resource inventory levels",
            "  - Respond to availability queries",
            "  - Approve or deny resource allocation requests",
        ])
        
        self.knowledge_base = "\n".join(kb_parts)
    
    def check_availability(self, resource: str, quantity: int) -> bool:
        """
        Check if the warehouse has sufficient quantity of a resource.
        
        ICDAM 2025 Table 1: Core availability check method.
        
        Args:
            resource: Resource identifier (e.g., 'R1', 'R2', 'R3', 'R4').
            quantity: Required quantity.
            
        Returns:
            True if inventory >= quantity, False otherwise.
        """
        available = self.inventory.get(resource, 0)
        return available >= quantity
    
    def get_inventory_status(self) -> Dict[str, int]:
        """
        Get current inventory levels.
        
        Returns:
            Copy of the inventory dictionary.
        """
        return dict(self.inventory)
    
    def allocate_resource(self, resource: str, quantity: int) -> bool:
        """
        Attempt to allocate (deduct) resources from inventory.
        
        Args:
            resource: Resource identifier.
            quantity: Amount to allocate.
            
        Returns:
            True if allocation succeeded, False if insufficient inventory.
        """
        if self.check_availability(resource, quantity):
            self.inventory[resource] -= quantity
            return True
        return False
    
    def release_resource(self, resource: str, quantity: int) -> None:
        """
        Release (return) resources back to inventory.
        
        Args:
            resource: Resource identifier.
            quantity: Amount to release.
        """
        if resource in self.inventory:
            self.inventory[resource] += quantity
        else:
            self.inventory[resource] = quantity
    
    def process_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming resource request using LLM-driven decision making.
        
        ICDAM 2025: Negotiation Layer - LLM evaluates request against inventory state.
        
        Args:
            message: Incoming negotiation message dict with keys:
                     'sender', 'receiver', 'content', 'type', 'timestamp'
                     
            Returns:
                Response message dict with LLM-generated content and decision type.
        """
        # Build system context with current state
        system_context = f"""You are a Warehouse Manager named {self.name}.
Your role is to manage resource inventory and respond to allocation requests.

CURRENT INVENTORY STATUS:
{self._format_inventory()}

DECISION RULES:
- If ALL requested resources are available in sufficient quantity: respond with "AGREED".
- If SOME resources are insufficient: respond with "COUNTER" and propose what you CAN provide.
- If NO resources are available: respond with "REJECT" and explain the shortage.

You must respond in a structured JSON format with the following fields:
- "thought": Your internal reasoning about the request and inventory.
- "speak": What you say to the other agent.
- "decision": One of "AGREED", "COUNTER", "REJECT".
- "proposal": (Optional) If decision is COUNTER, specify what you can offer.

Example:
{{
  "thought": "The PM requested 10 units of R1, but I only have 8. I can offer 8 now and the rest later.",
  "speak": "I'm sorry, I only have 8 units of R1 available right now. Would you like to take those?",
  "decision": "COUNTER",
  "proposal": {{"R1": 8}}
}}
"""

        # Get incoming request content
        request_content = message.get('content', '')
        
        # Use LLM to generate response
        llm_response_raw = self.brain.think_with_context(system_context, request_content)
        
        # Parse structured response
        parsed_response = JSONParser.parse_llm_response(llm_response_raw)
        
        decision = parsed_response.get("decision", "REJECT").upper()
        speak_content = parsed_response.get("speak", llm_response_raw)
        
        # Determine message type from decision
        if "AGREED" in decision:
            msg_type = "AGREE"
        elif "COUNTER" in decision:
            msg_type = "COUNTER"
        else:
            msg_type = "REJECT"
        
        # Construct response message
        response_message = {
            "sender": self.name,
            "receiver": message.get('sender', 'Unknown'),
            "content": speak_content,
            "thought": parsed_response.get("thought", ""),
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "in_response_to": message.get('timestamp', ''),
            "metadata": {
                "proposal": parsed_response.get("proposal", {})
            }
        }
        
        # Log in memory
        self.memory.append({
            **response_message,
            "direction": "sent"
        })
        
        return response_message
    
    def _format_inventory(self) -> str:
        """Format inventory for LLM context."""
        lines = []
        for resource, quantity in self.inventory.items():
            lines.append(f"  - {resource}: {quantity} units available")
        return "\n".join(lines)


class ProjectManagerAgent(BaseAgent):
    """
    State-aware Project Manager Agent for job scheduling with OR-Tools integration.
    
    ICDAM 2025 Table 1: Core MAS - ProjectManagerAgent Implementation.
    
    This agent maintains project job data loaded from benchmark files,
    calculates optimal baseline schedules using OR-Tools CP-SAT solver,
    and provides job information retrieval functionality.
    
    Attributes:
        project_data: Dictionary mapping job IDs to job details.
        optimal_makespan: Optimal project duration calculated by solver.
        optimal_schedule: Detailed schedule with job start/end times.
        state: General state dictionary for extensibility.
    """
    
    def __init__(
        self, 
        name: str, 
        project_data: Dict[int, Dict[str, Any]],
        resource_capacities: Optional[Dict[str, int]] = None,
        data_file_path: Optional[str] = None
    ):
        """
        Initialize ProjectManagerAgent with job data and compute optimal schedule.
        
        Args:
            name: Unique identifier for this agent.
            project_data: Dictionary of jobs from PSPLib loader.
                         Format: {job_id: {'duration': int, 'demands': dict, 'successors': list}}
            resource_capacities: Optional dictionary of global resource capacities.
            data_file_path: Optional path to PSPLib .sm file for solver optimization.
                           If provided, solver calculates optimal makespan at init.
        """
        super().__init__(name, role="Project Manager")
        
        # ICDAM Table 1: State-aware agent with project data
        self.project_data: Dict[int, Dict[str, Any]] = dict(project_data)
        self.resource_capacities: Dict[str, int] = dict(resource_capacities) if resource_capacities else {}
        
        # Store data file path for OptiMUS self-correction
        self.data_file_path: Optional[str] = data_file_path
        
        # Solver integration: optimal baseline
        self.optimal_makespan: Optional[int] = None
        self.optimal_schedule: Optional[Dict[int, int]] = None
        self.solver_status: str = "NOT_RUN"
        
        # OptiMUS: Store solver and parser for self-correction loop
        self.solver: Optional[RCPSPSolver] = None
        self.parser: Optional[RCPSPParser] = None
        self.solver_data: Optional[Dict[str, Any]] = None
        
        # Run solver if data file provided
        if data_file_path and os.path.exists(data_file_path):
            self._compute_optimal_baseline(data_file_path)
        
        self.state: Dict[str, Any] = {
            'initialized': True,
            'total_jobs': len(project_data),
            'current_job': None,
            'optimal_makespan': self.optimal_makespan,
            'solver_status': self.solver_status,
            'resource_capacities': self.resource_capacities
        }
        
        # Build knowledge base
        self._build_knowledge_base()
    
    def _compute_optimal_baseline(self, data_file_path: str) -> None:
        """
        Compute optimal baseline schedule using OR-Tools CP-SAT solver.
        
        ICDAM 2025: Integrates symbolic solver for optimal scheduling.
        OptiMUS: Stores solver/parser for self-correction loop.
        
        Args:
            data_file_path: Path to PSPLib .sm file.
        """
        try:
            # Parse the instance using solver's parser
            self.parser = RCPSPParser(data_file_path)
            self.solver_data = self.parser.parse()

            # Solve for optimal makespan
            self.solver = RCPSPSolver()
            makespan, status, schedule = self.solver.solve(self.solver_data, time_limit_seconds=60)

            self.optimal_makespan = makespan
            self.solver_status = status
            self.optimal_schedule = schedule

            # OptiMUS: Extract and store optimal schedule for grounding fallback
            if makespan is not None:
                self.optimal_schedule = self.solver.get_schedule_dict()

            # Log result
            if makespan is not None:
                print(f"[{self.name}] Calculated Optimal Baseline: {makespan} days ({status})")
            else:
                print(f"[{self.name}] Solver returned no solution ({status})")

        except Exception as e:
            print(f"[{self.name}] Solver error: {type(e).__name__}: {e}")
            self.solver_status = "ERROR"
    
    def _build_knowledge_base(self) -> None:
        """Construct agent's knowledge base from project data."""
        kb_parts = [
            f"ROLE: {self.role}",
            f"NAME: {self.name}",
            "",
            f"PROJECT OVERVIEW:",
            f"  - Total Jobs: {len(self.project_data)}",
        ]
        
        # Add solver result if available
        if self.optimal_makespan is not None:
            kb_parts.append(f"  - Optimal Makespan: {self.optimal_makespan} days ({self.solver_status})")
        
        kb_parts.append("")
        kb_parts.append("JOB SUMMARY (first 5 non-dummy jobs):")
        
        # Show first 5 non-dummy jobs (job_id > 1, excluding source/sink)
        shown = 0
        for job_id in sorted(self.project_data.keys()):
            if job_id > 1 and shown < 5:
                job = self.project_data[job_id]
                duration = job.get('duration', 0)
                if duration > 0:  # Skip dummy jobs with 0 duration
                    kb_parts.append(f"  - Job {job_id}: Duration={duration}, Successors={job.get('successors', [])}")
                    shown += 1
        
        kb_parts.extend([
            "",
            "RESPONSIBILITIES:",
            "  - Schedule jobs respecting precedence constraints",
            "  - Coordinate resource requests with Warehouse",
            "  - Minimize project makespan",
        ])
        
        self.knowledge_base = "\n".join(kb_parts)
    
    def get_job_details(self, job_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific job.
        
        ICDAM 2025 Table 1: Core job information retrieval method.
        
        Args:
            job_id: Job identifier (1-indexed as per PSPLib format).
            
        Returns:
            Dictionary with 'duration', 'demands', and 'successors',
            or None if job_id not found.
        """
        job = self.project_data.get(job_id)
        if job is None:
            return None
        
        return {
            'duration': job.get('duration', 0),
            'demands': job.get('demands', {}),
            'successors': job.get('successors', [])
        }
    
    def get_all_job_ids(self) -> List[int]:
        """
        Get list of all job IDs in the project.
        
        Returns:
            Sorted list of job IDs.
        """
        return sorted(self.project_data.keys())
    
    def get_job_successors(self, job_id: int) -> List[int]:
        """
        Get successor jobs for a given job (precedence constraint).
        
        Args:
            job_id: Job identifier.
            
        Returns:
            List of successor job IDs, empty list if not found.
        """
        job = self.project_data.get(job_id)
        if job is None:
            return []
        return job.get('successors', [])
    
    def set_current_job(self, job_id: int) -> bool:
        """
        Set the currently active job for scheduling.
        
        Args:
            job_id: Job identifier to set as current.
            
        Returns:
            True if job exists and was set, False otherwise.
        """
        if job_id in self.project_data:
            self.state['current_job'] = job_id
            return True
        return False
    
    def request_resources(
        self, 
        job_id: int, 
        warehouse: 'WarehouseAgent'
    ) -> Dict[str, Any]:
        """
        Generate and send a resource request for a specific job using LLM.
        
        ICDAM 2025: Negotiation Layer - LLM generates formal resource proposals
        with solver-computed optimal baseline context.
        
        Args:
            job_id: Job identifier to request resources for.
            warehouse: Target WarehouseAgent to send request to.
            
        Returns:
            Dictionary containing the sent message, or error info if job not found.
        """
        # Get job details
        job_details = self.get_job_details(job_id)
        
        if job_details is None:
            return {
                "error": True,
                "message": f"Job {job_id} not found in project data."
            }
        
        # Format demands for the prompt
        demands_str = ", ".join([
            f"{res}: {qty} units" 
            for res, qty in job_details['demands'].items() 
            if qty > 0
        ])
        
        if not demands_str:
            demands_str = "no resources (dummy job)"
        
        # Include optimal makespan context if available
        makespan_context = ""
        if self.optimal_makespan is not None:
            makespan_context = f"""
SOLVER BASELINE:
- The OR-Tools solver estimates the total project should take {self.optimal_makespan} days.
- This is the optimal schedule respecting all constraints.
"""
        
        # Construct prompt for LLM to generate formal proposal
        prompt = f"""You are a Project Manager named {self.name}.
You need to secure resources for Job #{job_id}.

JOB DETAILS:
- Duration: {job_details['duration']} time units
- Resource Requirements: {demands_str}
- Successor Jobs: {job_details['successors']}
{makespan_context}

Write a formal PROPOSAL message to the Warehouse Manager requesting these resources.
Respond in a structured JSON format:
{{
  "thought": "Your internal reasoning about the project timeline and resource needs.",
  "speak": "The message you want to send to the Warehouse Manager.",
  "function": "PROPOSE"
}}
"""

        # Use LLM to generate the proposal
        llm_response_raw = self.brain.think(prompt)
        
        # Parse structured response
        parsed_response = JSONParser.parse_llm_response(llm_response_raw)
        proposal_content = parsed_response.get("speak", llm_response_raw)
        
        # Update current job state
        self.state['current_job'] = job_id
        
        # Send negotiation message to warehouse
        message = self.send_negotiation(
            receiver=warehouse,
            content=proposal_content,
            msg_type="PROPOSE"
        )
        
        # Add thought to message
        message["thought"] = parsed_response.get("thought", "")
        
        return message

    def develop_viable_plan(
        self, 
        max_retries: int = 3
    ) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        OptiMUS Self-Correction Loop: Iteratively refine schedule using LLM + Solver feedback.
        
        This method implements the core OptiMUS pattern:
        1. LLM generates initial schedule proposal
        2. Solver verifies feasibility
        3. If infeasible, error feedback sent to LLM for repair
        4. Loop until feasible or max retries reached
        5. Fallback to optimal solver schedule (Grounding mechanism)
        
        ICDAM 2025: Key contribution - hybrid LLM reasoning with symbolic verification.
        
        Args:
            max_retries: Maximum number of LLM repair attempts before fallback.
            
        Returns:
            Tuple of (schedule_dict, result_info):
            - schedule_dict: Valid schedule {job_id: start_time}
            - result_info: Metadata about the process
        """
        result_info = {
            'method': 'llm',  # 'llm' or 'solver_fallback'
            'attempts': 0,
            'history': [],
            'final_makespan': None,
            'success': False
        }
        
        # Check prerequisites
        if self.solver is None or self.solver_data is None:
            print(f"[{self.name}] ERROR: Solver not initialized. Cannot develop plan.")
            if self.optimal_schedule:
                result_info['method'] = 'solver_fallback'
                result_info['success'] = True
                result_info['final_makespan'] = self.optimal_makespan
                return self.optimal_schedule, result_info
            return {}, result_info
        
        # Build project context for LLM
        project_context = self._build_scheduling_context()
        
        # Step 1: Ask LLM to generate initial schedule
        print(f"\n[{self.name}] OptiMUS Self-Correction Loop Started (max_retries={max_retries})")
        print("-" * 60)
        
        initial_prompt = f"""You are a Project Manager creating a schedule for an RCPSP project.

{project_context}

TASK: Create a feasible schedule that respects ALL precedence constraints.
- A job can only start AFTER all its predecessors have completed.
- Format your answer as: Job 1: Start 0, Job 2: Start X, Job 3: Start Y, ...
- Include ALL {len(self.project_data)} jobs in your schedule.
- Job 1 (dummy start) should start at time 0.
- Job {len(self.project_data)} (dummy end) should start after all predecessors complete.

Provide the complete schedule now:"""

        current_prompt = initial_prompt
        schedule = {}
        
        for attempt in range(1, max_retries + 1):
            result_info['attempts'] = attempt
            print(f"\n[Attempt {attempt}/{max_retries}] Asking LLM to generate schedule...")
            
            # Get LLM response
            llm_response = self.brain.think(current_prompt)
            
            # Log attempt
            attempt_log = {
                'attempt': attempt,
                'prompt_preview': current_prompt[:200] + "...",
                'response_preview': llm_response[:200] + "...",
                'parsed_schedule': None,
                'verification': None
            }
            
            print(f"[Attempt {attempt}] LLM Response: {llm_response[:100]}...")
            
            # Step 2: Parse LLM output
            schedule = parse_schedule_from_text(llm_response)
            attempt_log['parsed_schedule'] = schedule
            
            if not schedule:
                print(f"[Attempt {attempt}] FAILED: Could not parse schedule from LLM output")
                
                # Create repair prompt for parsing failure
                current_prompt = f"""Your previous response could not be parsed into a schedule.

Please provide the schedule in this EXACT format:
Job 1: Start 0, Job 2: Start 3, Job 3: Start 5, Job 4: Start 8, ...

Include ALL {len(self.project_data)} jobs. Each job needs "Job X: Start Y" format.
Try again:"""
                
                result_info['history'].append(attempt_log)
                continue
            
            print(f"[Attempt {attempt}] Parsed {len(schedule)} jobs from LLM output")
            
            # Step 3: Verify with solver
            verification = self.solver.verify_schedule(self.solver_data, schedule)
            attempt_log['verification'] = {
                'is_feasible': verification['is_feasible'],
                'error_preview': verification['error_message'][:200] if verification['error_message'] else '',
                'violation_count': len(verification['violations'])
            }
            
            if verification['is_feasible']:
                # SUCCESS! Schedule is valid
                print(f"[Attempt {attempt}] SUCCESS! Schedule is feasible. Makespan: {verification['makespan']}")
                result_info['success'] = True
                result_info['method'] = 'llm'
                result_info['final_makespan'] = verification['makespan']
                result_info['history'].append(attempt_log)
                
                return schedule, result_info
            
            else:
                # FAILED: Create repair prompt with error feedback
                error_msg = verification['error_message']
                print(f"[Attempt {attempt}] FAILED: {error_msg[:100]}...")
                
                # Build specific repair prompt based on violation type
                violations = verification['violations']
                
                repair_prompt = f"""Your proposed schedule is INVALID.

ERROR DETAILS:
{error_msg}

VIOLATION SUMMARY:
- Precedence violations: {len([v for v in violations if v['type'] == 'PRECEDENCE_VIOLATION'])}
- Capacity violations: {len([v for v in violations if v['type'] == 'CAPACITY_VIOLATION'])}
- Missing jobs: {len([v for v in violations if v['type'] == 'MISSING_JOBS'])}

{project_context}

REPAIR INSTRUCTIONS:
1. Review the precedence constraints carefully.
2. A successor job MUST start at or after its predecessor ENDS (start + duration).
3. Fix the violations mentioned above.

Provide the CORRECTED schedule in format: Job 1: Start 0, Job 2: Start X, ...
Include ALL {len(self.project_data)} jobs:"""

                current_prompt = repair_prompt
                result_info['history'].append(attempt_log)
        
        # Step 4: Fallback to optimal solver schedule (Grounding mechanism)
        print(f"\n[{self.name}] LLM failed after {max_retries} attempts. Falling back to OR-Tools optimal schedule.")
        
        if self.optimal_schedule:
            result_info['method'] = 'solver_fallback'
            result_info['success'] = True
            result_info['final_makespan'] = self.optimal_makespan
            print(f"[{self.name}] Using optimal schedule with makespan: {self.optimal_makespan}")
            return self.optimal_schedule, result_info
        else:
            print(f"[{self.name}] ERROR: No fallback schedule available!")
            return {}, result_info
    
    def _build_scheduling_context(self) -> str:
        """
        Build project context string for LLM scheduling prompts.
        
        Returns:
            Formatted string with job details and constraints.
        """
        lines = [
            f"PROJECT: {len(self.project_data)} jobs total",
            f"OPTIMAL BASELINE: {self.optimal_makespan} time units (computed by OR-Tools)",
            "",
            "JOB DETAILS (job_id: duration, predecessors):"
        ]
        
        # Build predecessor map from successor data
        predecessors: Dict[int, List[int]] = {job_id: [] for job_id in self.project_data.keys()}
        for job_id, job in self.project_data.items():
            for succ in job.get('successors', []):
                if succ in predecessors:
                    predecessors[succ].append(job_id)
        
        # List all jobs with their constraints
        for job_id in sorted(self.project_data.keys()):
            job = self.project_data[job_id]
            duration = job.get('duration', 0)
            preds = predecessors.get(job_id, [])
            
            if preds:
                preds_str = f", must start after jobs {preds} complete"
            else:
                preds_str = ", no predecessors (can start at time 0)"
            
            lines.append(f"  Job {job_id}: duration={duration}{preds_str}")
        
        lines.append("")
        lines.append("PRECEDENCE RULE: Job X can start only when ALL its predecessors have finished.")
        lines.append("                 (predecessor_start + predecessor_duration <= job_start)")
        
        return "\n".join(lines)


# =============================================================================
# Legacy Agents (Backward Compatibility)
# =============================================================================


class ResourceAgent(BaseAgent):
    """
    Specialized agent for managing resources.
    Handles resource allocation, availability tracking, and conflict resolution.
    """
    
    def __init__(self, name, resource_data):
        """
        Initialize Resource Agent.
        
        Args:
            name: Agent name
            resource_data: Dictionary containing resource view from RCPSPScenario
        """
        super().__init__(name, role="Resource Manager")
        self.resource_data = resource_data
        
        # Build knowledge base from resource data
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Construct the agent's knowledge base from resource data."""
        kb_parts = []
        
        # Add role description
        kb_parts.append(f"ROLE: {self.role}")
        kb_parts.append(f"NAME: {self.name}")
        kb_parts.append("")
        
        # Add resource summary
        kb_parts.append("RESOURCE INFORMATION:")
        kb_parts.append(self.resource_data['summary'])
        kb_parts.append("")
        
        # Add detailed resource descriptions
        kb_parts.append("AVAILABLE RESOURCES:")
        for res in self.resource_data['resources']:
            kb_parts.append(f"  - {res['description']}")
        
        kb_parts.append("")
        kb_parts.append("RESPONSIBILITIES:")
        kb_parts.append("  - Monitor resource availability")
        kb_parts.append("  - Approve or deny resource allocation requests")
        kb_parts.append("  - Track resource usage across tasks")
        kb_parts.append("  - Prevent resource over-allocation")
        
        self.knowledge_base = "\n".join(kb_parts)
    
    def get_resource_capacity(self, resource_name):
        """
        Get the capacity of a specific resource.
        
        Args:
            resource_name: Name of the resource (e.g., "R1")
            
        Returns:
            int: Resource capacity, or None if not found
        """
        return self.resource_data['capacities'].get(resource_name)
    
    def think(self):
        """
        Override think method with resource-specific logic.
        
        Returns:
            str: Agent's reasoning about resource allocation
        """
        base_thought = super().think()
        
        # Add resource-specific context
        received_messages = [m for m in self.memory if m['type'] == 'received']
        if received_messages:
            last_message = received_messages[-1]['content']
            
            # Mock logic: Check if message is about resource booking
            if "book" in last_message.lower() or "allocate" in last_message.lower():
                resource_thought = (f"\n[{self.name}] Resource Analysis: "
                                  f"Checking availability against capacities: "
                                  f"{self.resource_data['capacities']}")
                return base_thought + resource_thought
        
        return base_thought


class LegacyProjectManagerAgent(BaseAgent):
    """
    Legacy Project Manager Agent (for backward compatibility with RCPSPScenario).
    
    NOTE: For new code, use ProjectManagerAgent which accepts PSPLib loader data.
    
    Specialized agent for managing project tasks and scheduling.
    Handles task allocation, precedence constraints, and timeline optimization.
    """
    
    def __init__(self, name, task_data):
        """
        Initialize Legacy Project Manager Agent.
        
        Args:
            name: Agent name
            task_data: Dictionary containing task view from RCPSPScenario
        """
        super().__init__(name, role="Project Manager")
        self.task_data = task_data
        
        # Build knowledge base from task data
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Construct the agent's knowledge base from task data."""
        kb_parts = []
        
        # Add role description
        kb_parts.append(f"ROLE: {self.role}")
        kb_parts.append(f"NAME: {self.name}")
        kb_parts.append("")
        
        # Add task summary
        kb_parts.append("PROJECT INFORMATION:")
        kb_parts.append(self.task_data['summary'])
        kb_parts.append(f"Non-dummy tasks: {self.task_data['total_tasks']}")
        kb_parts.append("")
        
        # Add task descriptions (sample first 5 to avoid overwhelming the KB)
        kb_parts.append("TASK DETAILS (sample):")
        for task in self.task_data['tasks'][:5]:
            kb_parts.append(f"  - {task['description']}")
        
        if len(self.task_data['tasks']) > 5:
            kb_parts.append(f"  ... and {len(self.task_data['tasks']) - 5} more tasks")
        
        kb_parts.append("")
        kb_parts.append("RESPONSIBILITIES:")
        kb_parts.append("  - Schedule tasks to minimize project duration")
        kb_parts.append("  - Ensure precedence constraints are satisfied")
        kb_parts.append("  - Coordinate with Resource Managers for allocation")
        kb_parts.append("  - Optimize resource utilization")
        
        self.knowledge_base = "\n".join(kb_parts)
    
    def get_task_info(self, job_id):
        """
        Get information about a specific task.
        
        Args:
            job_id: Job ID (1-indexed)
            
        Returns:
            dict: Task information, or None if not found
        """
        for task in self.task_data['tasks']:
            if task['job_id'] == job_id:
                return task
        return None
    
    def think(self):
        """
        Override think method with project management-specific logic.
        
        Returns:
            str: Agent's reasoning about scheduling decisions
        """
        base_thought = super().think()
        
        # Add PM-specific context
        pm_thought = (f"\n[{self.name}] Scheduling Analysis: "
                     f"Managing {self.task_data['total_tasks']} tasks with precedence constraints")
        
        return base_thought + pm_thought


def main():
    """Test the agent framework with a sample scenario (legacy mode)."""
    print("=" * 70)
    print("MULTI-AGENT SYSTEM - Basic Agent Framework Test (Legacy)")
    print("=" * 70)
    print("\nNOTE: For new state-aware agents, run main_simulation.py instead.\n")
    
    # Check if RCPSPScenario is available
    if RCPSPScenario is None:
        print("[ERROR] RCPSPScenario not available. Use main_simulation.py instead.")
        return
    
    # Initialize scenario
    instance_file = os.path.join("data", "raw", "rcpsp", "j30", "j3010_1.sm")
    
    if not os.path.exists(instance_file):
        print(f"\n Error: File not found: {instance_file}")
        return
    
    print(f"\n[1] Loading scenario: {instance_file}")
    scenario = RCPSPScenario(instance_file)
    
    # Get views
    resource_view = scenario.get_resource_view()
    task_view = scenario.get_task_view()
    
    print(f" Loaded scenario with {scenario.n_jobs} jobs and {scenario.n_resources} resources")
    
    # Create agents (using legacy agents for backward compatibility)
    print(f"\n{'=' * 70}")
    print("[2] Creating Agents (Legacy Mode)")
    print("=" * 70)
    
    warehouse_agent = ResourceAgent("Warehouse_A", resource_view)
    pm_agent = LegacyProjectManagerAgent("PM_B", task_view)
    
    print(f"\n Created {warehouse_agent.name} ({warehouse_agent.role})")
    print(f" Created {pm_agent.name} ({pm_agent.role})")
    
    # Display knowledge bases
    print(f"\n{'=' * 70}")
    print("[3] Agent Knowledge Bases")
    print("=" * 70)
    
    print(f"\n{warehouse_agent.name} Knowledge Base:")
    print("-" * 70)
    print(warehouse_agent.knowledge_base)
    
    print(f"\n{pm_agent.name} Knowledge Base:")
    print("-" * 70)
    print(pm_agent.knowledge_base)
    
    # Simulate interaction
    print(f"\n{'=' * 70}")
    print("[4] Agent Interaction Simulation")
    print("=" * 70)
    
    print("\n--- Scenario: PM requests resource allocation ---\n")
    
    # PM sends a message
    message_content = "Can I book 2 units of R1 for Job 2?"
    pm_agent.send_message(warehouse_agent.name, message_content)
    
    # Warehouse receives the message
    warehouse_agent.receive_message(pm_agent.name, message_content)
    
    # Warehouse thinks and responds
    print("\n--- Resource Agent Thinking ---")
    warehouse_response = warehouse_agent.think()
    print(warehouse_response)
    
    # Display agent contexts
    print(f"\n{'=' * 70}")
    print("[5] Agent States")
    print("=" * 70)
    
    print(f"\n{warehouse_agent.name} Context:")
    warehouse_context = warehouse_agent.get_context()
    print(f"  - Role: {warehouse_context['role']}")
    print(f"  - Memory Size: {warehouse_context['memory_size']} messages")
    print(f"  - Recent Messages: {len(warehouse_context['recent_messages'])}")
    
    print(f"\n{pm_agent.name} Context:")
    pm_context = pm_agent.get_context()
    print(f"  - Role: {pm_context['role']}")
    print(f"  - Memory Size: {pm_context['memory_size']} messages")
    print(f"  - Recent Messages: {len(pm_context['recent_messages'])}")
    
    print(f"\n{'=' * 70}")
    print("AGENT FRAMEWORK TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
