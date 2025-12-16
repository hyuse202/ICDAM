"""
Basic Agent Framework for Multi-Agent System
Defines base agent structure and specialized agents for RCPSP problem solving.
"""

import os
from datetime import datetime
from scenarios.rcpsp_scenario import RCPSPScenario


class BaseAgent:
    """
    Base class for all agents in the Multi-Agent System.
    Provides core functionality for message handling and reasoning.
    """
    
    def __init__(self, name, role):
        """
        Initialize the base agent.
        
        Args:
            name: Unique identifier for the agent
            role: Role description (e.g., "Resource Manager", "Project Manager")
        """
        self.name = name
        self.role = role
        self.memory = []  # List of message dictionaries
        self.knowledge_base = ""  # Context/system prompt
        
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


class ProjectManagerAgent(BaseAgent):
    """
    Specialized agent for managing project tasks and scheduling.
    Handles task allocation, precedence constraints, and timeline optimization.
    """
    
    def __init__(self, name, task_data):
        """
        Initialize Project Manager Agent.
        
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
    """Test the agent framework with a sample scenario."""
    print("=" * 70)
    print("MULTI-AGENT SYSTEM - Basic Agent Framework Test")
    print("=" * 70)
    
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
    
    # Create agents
    print(f"\n{'=' * 70}")
    print("[2] Creating Agents")
    print("=" * 70)
    
    warehouse_agent = ResourceAgent("Warehouse_A", resource_view)
    pm_agent = ProjectManagerAgent("PM_B", task_view)
    
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
