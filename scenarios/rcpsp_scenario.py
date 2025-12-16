"""
RCPSP Scenario Module
Bridges raw RCPSP data and LLM agents by providing natural language views.
"""

import os
import json
from solvers.rcpsp_solver import RCPSPParser


class RCPSPScenario:
    """
    Represents an RCPSP instance with LLM-friendly views.
    Provides structured context for different agent roles.
    """
    
    def __init__(self, filepath):
        """
        Initialize scenario from a PSPLib .sm file.
        
        Args:
            filepath: Path to the .sm instance file
        """
        self.filepath = filepath
        self.instance_name = os.path.basename(filepath)
        
        # Parse the instance
        parser = RCPSPParser(filepath)
        self.data = parser.parse()
        
        # Extract key information
        self.n_jobs = self.data['n_jobs']
        self.n_resources = self.data['n_resources']
        self.durations = self.data['durations']
        self.resource_requirements = self.data['resource_requirements']
        self.successors = self.data['successors']
        self.resource_capacities = self.data['resource_capacities']
    
    def get_resource_view(self):
        """
        Generate a natural language view of resources for the Resource Manager Agent.
        
        Returns:
            dict: Resource information in LLM-friendly format
                - summary: Overall resource description
                - resources: List of resource descriptions
                - capacities: Dictionary of resource capacities
        """
        resource_view = {
            'summary': f"Project has {self.n_resources} renewable resources with limited capacities.",
            'resources': [],
            'capacities': {}
        }
        
        # Generate descriptions for each resource
        for i in range(self.n_resources):
            resource_name = f"R{i+1}"
            capacity = self.resource_capacities[i]
            
            description = f"Renewable Resource {resource_name} has capacity {capacity} units."
            resource_view['resources'].append({
                'name': resource_name,
                'capacity': capacity,
                'description': description
            })
            resource_view['capacities'][resource_name] = capacity
        
        return resource_view
    
    def get_task_view(self):
        """
        Generate a task-oriented view for the Project Manager Agent.
        
        Returns:
            dict: Task information including:
                - summary: Overall project description
                - tasks: List of task dictionaries with details
                - total_tasks: Number of tasks (excluding dummy nodes)
        """
        task_view = {
            'summary': f"Project consists of {self.n_jobs} jobs (including start and end nodes).",
            'tasks': [],
            'total_tasks': self.n_jobs - 2  # Exclude dummy start and end
        }
        
        # Generate task information
        for job_id in range(self.n_jobs):
            task_info = {
                'job_id': job_id + 1,  # 1-indexed for readability
                'duration': self.durations[job_id],
                'resource_usage': {},
                'successors': [s + 1 for s in self.successors[job_id]],  # Convert to 1-indexed
                'is_dummy': job_id == 0 or job_id == self.n_jobs - 1
            }
            
            # Add resource requirements
            for res_idx in range(self.n_resources):
                resource_name = f"R{res_idx + 1}"
                usage = self.resource_requirements[job_id][res_idx]
                if usage > 0:
                    task_info['resource_usage'][resource_name] = usage
            
            # Generate natural language description
            if task_info['is_dummy']:
                if job_id == 0:
                    description = "Job 1: Project START (dummy task, duration 0)"
                else:
                    description = f"Job {self.n_jobs}: Project END (dummy task, duration 0)"
            else:
                res_desc = ", ".join([f"{k}={v}" for k, v in task_info['resource_usage'].items()])
                if not res_desc:
                    res_desc = "no resources"
                
                succ_desc = ", ".join([str(s) for s in task_info['successors']])
                if not succ_desc:
                    succ_desc = "none"
                
                description = (f"Job {task_info['job_id']}: Duration {task_info['duration']} time units, "
                              f"requires [{res_desc}], must finish before jobs [{succ_desc}]")
            
            task_info['description'] = description
            task_view['tasks'].append(task_info)
        
        return task_view
    
    def get_precedence_view(self):
        """
        Generate a precedence-focused view showing all predecessor-successor relationships.
        
        Returns:
            dict: Precedence information
        """
        precedence_view = {
            'summary': "Task precedence constraints (predecessor -> successor relationships)",
            'relationships': []
        }
        
        for job_id in range(self.n_jobs):
            for succ_id in self.successors[job_id]:
                precedence_view['relationships'].append({
                    'predecessor': job_id + 1,
                    'successor': succ_id + 1,
                    'description': f"Job {job_id + 1} must complete before Job {succ_id + 1} can start"
                })
        
        return precedence_view
    
    def to_json(self):
        """
        Export the entire scenario to a JSON-serializable dictionary.
        
        Returns:
            dict: Complete scenario data in JSON format
        """
        return {
            'instance_name': self.instance_name,
            'filepath': self.filepath,
            'project_info': {
                'n_jobs': self.n_jobs,
                'n_resources': self.n_resources,
                'total_tasks': self.n_jobs - 2
            },
            'resources': {
                'capacities': self.resource_capacities,
                'count': self.n_resources
            },
            'tasks': [
                {
                    'job_id': i + 1,
                    'duration': self.durations[i],
                    'resource_requirements': {
                        f"R{j+1}": self.resource_requirements[i][j] 
                        for j in range(self.n_resources)
                    },
                    'successors': [s + 1 for s in self.successors[i]]
                }
                for i in range(self.n_jobs)
            ],
            'views': {
                'resource_view': self.get_resource_view(),
                'task_view': self.get_task_view(),
                'precedence_view': self.get_precedence_view()
            }
        }
    
    def export_to_file(self, output_path):
        """
        Export scenario to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        data = self.to_json()
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Scenario exported to: {output_path}")


def main():
    """Test the scenario module with a sample instance."""
    print("=" * 70)
    print("RCPSP SCENARIO - Agent View Generator")
    print("=" * 70)
    
    # Load instance
    instance_file = os.path.join("data", "raw", "rcpsp", "j30", "j3010_1.sm")
    
    if not os.path.exists(instance_file):
        print(f"\n✗ Error: File not found: {instance_file}")
        return
    
    print(f"\n[1] Loading instance: {instance_file}")
    scenario = RCPSPScenario(instance_file)
    print(f"    Instance: {scenario.instance_name}")
    print(f"    Jobs: {scenario.n_jobs}")
    print(f"    Resources: {scenario.n_resources}")
    
    # Resource View
    print(f"\n{'=' * 70}")
    print("[2] RESOURCE VIEW (for Resource Manager Agent)")
    print("=" * 70)
    resource_view = scenario.get_resource_view()
    print(f"\nSummary: {resource_view['summary']}\n")
    for res in resource_view['resources']:
        print(f"  • {res['description']}")
    
    # Task View
    print(f"\n{'=' * 70}")
    print("[3] TASK VIEW (for Project Manager Agent)")
    print("=" * 70)
    task_view = scenario.get_task_view()
    print(f"\nSummary: {task_view['summary']}")
    print(f"Total non-dummy tasks: {task_view['total_tasks']}\n")
    
    print("Task Details:")
    for task in task_view['tasks'][:10]:  # Show first 10 tasks
        print(f"  • {task['description']}")
    
    if len(task_view['tasks']) > 10:
        print(f"  ... (and {len(task_view['tasks']) - 10} more tasks)")
    
    # JSON Export
    print(f"\n{'=' * 70}")
    print("[4] JSON EXPORT")
    print("=" * 70)
    json_data = scenario.to_json()
    print(f"JSON structure includes:")
    print(f"  - instance_name: {json_data['instance_name']}")
    print(f"  - project_info: {json_data['project_info']}")
    print(f"  - resources: {len(json_data['resources']['capacities'])} resources")
    print(f"  - tasks: {len(json_data['tasks'])} tasks")
    print(f"  - views: {len(json_data['views'])} agent views")
    
    print(f"\n{'=' * 70}")
    print("SCENARIO MODULE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
