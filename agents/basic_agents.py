"""
Basic Agent Framework for Multi-Agent System
ICDAM 2025 - Table 1: Core MAS Infrastructure + Negotiation Layer

Defines base agent structure and specialized agents for RCPSP problem solving.
Implements state-aware agents following AgentScope patterns with LLM-driven negotiation.

References:
- AgentScope: AgentBase class and Message Passing architecture
- REALM-Bench: Benchmark-driven agent evaluation
- ICDAM 2025: Hybrid LLM Multi-Agent System for SCM
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import LLMBrain for negotiation layer
from agents.llm_brain import LLMBrain

# Import Solver components for optimal scheduling
from solvers.rcpsp_solver import RCPSPSolver, RCPSPParser

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
- If ALL requested resources are available in sufficient quantity: respond with "AGREED" and confirm allocation.
- If SOME resources are insufficient: respond with "COUNTER" and propose what you CAN provide.
- If NO resources are available: respond with "REJECT" and explain the shortage.

Keep your response concise and professional (2-3 sentences max).
Start your response with the decision word: AGREED, COUNTER, or REJECT."""

        # Get incoming request content
        request_content = message.get('content', '')
        
        # Use LLM to generate response
        llm_response = self.brain.think_with_context(system_context, request_content)
        
        # Determine message type from LLM response
        response_upper = llm_response.upper()
        if "AGREED" in response_upper:
            msg_type = "AGREE"
        elif "COUNTER" in response_upper:
            msg_type = "COUNTER"
        else:
            msg_type = "REJECT"
        
        # Construct response message
        response_message = {
            "sender": self.name,
            "receiver": message.get('sender', 'Unknown'),
            "content": llm_response,
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "in_response_to": message.get('timestamp', '')
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
        data_file_path: Optional[str] = None
    ):
        """
        Initialize ProjectManagerAgent with job data and compute optimal schedule.
        
        Args:
            name: Unique identifier for this agent.
            project_data: Dictionary of jobs from PSPLib loader.
                         Format: {job_id: {'duration': int, 'demands': dict, 'successors': list}}
            data_file_path: Optional path to PSPLib .sm file for solver optimization.
                           If provided, solver calculates optimal makespan at init.
        """
        super().__init__(name, role="Project Manager")
        
        # ICDAM Table 1: State-aware agent with project data
        self.project_data: Dict[int, Dict[str, Any]] = dict(project_data)
        
        # Solver integration: optimal baseline
        self.optimal_makespan: Optional[int] = None
        self.optimal_schedule: Optional[Dict[str, Any]] = None
        self.solver_status: str = "NOT_RUN"
        
        # Run solver if data file provided
        if data_file_path and os.path.exists(data_file_path):
            self._compute_optimal_baseline(data_file_path)
        
        self.state: Dict[str, Any] = {
            'initialized': True,
            'total_jobs': len(project_data),
            'current_job': None,
            'optimal_makespan': self.optimal_makespan,
            'solver_status': self.solver_status
        }
        
        # Build knowledge base
        self._build_knowledge_base()
    
    def _compute_optimal_baseline(self, data_file_path: str) -> None:
        """
        Compute optimal baseline schedule using OR-Tools CP-SAT solver.
        
        ICDAM 2025: Integrates symbolic solver for optimal scheduling.
        
        Args:
            data_file_path: Path to PSPLib .sm file.
        """
        try:
            # Parse the instance using solver's parser
            parser = RCPSPParser(data_file_path)
            solver_data = parser.parse()
            
            # Solve for optimal makespan
            solver = RCPSPSolver()
            makespan, status = solver.solve(solver_data, time_limit_seconds=60)
            
            self.optimal_makespan = makespan
            self.solver_status = status
            
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
Be professional, concise (2-3 sentences), and clearly state what you need.
Reference the project timeline if solver baseline is available.
Start with "PROPOSAL:" followed by your request."""

        # Use LLM to generate the proposal
        proposal_content = self.brain.think(prompt)
        
        # Update current job state
        self.state['current_job'] = job_id
        
        # Send negotiation message to warehouse
        message = self.send_negotiation(
            receiver=warehouse,
            content=proposal_content,
            msg_type="PROPOSE"
        )
        
        return message


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
