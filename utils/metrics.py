"""
Simulation Metrics Module
ICDAM 2025 - Table 3: Performance Evaluation Metrics

This module implements key KPIs for evaluating the Hybrid LLM Multi-Agent System:
- Feasibility Rate: Percentage of valid schedules generated
- Cost Gap: Deviation from optimal solution (makespan)
- Negotiation Efficiency: Rounds needed to reach agreement
- Self-Correction Statistics: LLM repair attempts and success rate

References:
- REALM-Bench: Benchmark metrics for LLM reasoning evaluation
- Self-Resource Paper: Resource allocation performance metrics
- ICDAM 2025: Table 3 - Metrics (Thước Đo Hiệu Suất)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import statistics


@dataclass
class RunRecord:
    """Record of a single simulation run."""
    run_id: int
    timestamp: str
    instance_name: str
    
    # Schedule metrics
    agent_makespan: int
    optimal_makespan: int
    gap: float  # (agent - optimal) / optimal
    gap_percent: float
    
    # Feasibility
    is_feasible: bool
    method_used: str  # 'llm' or 'solver_fallback'
    
    # Self-correction loop
    correction_attempts: int
    llm_success: bool  # True if LLM produced valid schedule
    
    # Negotiation metrics
    negotiation_rounds: int
    messages_exchanged: int
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulationMetrics:
    """
    Collects and analyzes performance metrics for the MAS simulation.
    
    ICDAM 2025 Table 3: Implements all required performance indicators.
    
    Metrics Categories:
    1. Feasibility Metrics: Success rate of generating valid schedules
    2. Optimality Metrics: Gap between agent and optimal solutions
    3. Efficiency Metrics: Negotiation rounds, correction attempts
    4. LLM Performance: Self-correction success rate
    
    Attributes:
        runs: List of RunRecord objects containing all run data.
        start_time: Timestamp when metrics collection started.
    """
    
    def __init__(self, experiment_name: str = "ICDAM_Simulation"):
        """
        Initialize the metrics collector.
        
        Args:
            experiment_name: Name of the experiment for reporting.
        """
        self.experiment_name = experiment_name
        self.runs: List[RunRecord] = []
        self.start_time = datetime.now().isoformat()
        self._run_counter = 0
    
    def record_run(
        self,
        agent_makespan: int,
        optimal_makespan: int,
        is_feasible: bool,
        negotiation_rounds: int = 0,
        instance_name: str = "unknown",
        method_used: str = "llm",
        correction_attempts: int = 0,
        messages_exchanged: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RunRecord:
        """
        Record metrics for a single simulation run.
        
        ICDAM 2025: Core method for collecting Table 3 metrics.
        
        Args:
            agent_makespan: Makespan achieved by agent (LLM or fallback).
            optimal_makespan: Optimal makespan from OR-Tools solver.
            is_feasible: Whether the final schedule is feasible.
            negotiation_rounds: Number of negotiation rounds with other agents.
            instance_name: Name of the benchmark instance.
            method_used: 'llm' if LLM succeeded, 'solver_fallback' otherwise.
            correction_attempts: Number of self-correction attempts.
            messages_exchanged: Total messages exchanged between agents.
            metadata: Additional run-specific data.
            
        Returns:
            RunRecord object with all metrics for this run.
        """
        self._run_counter += 1
        
        # Calculate gap (handle division by zero)
        if optimal_makespan and optimal_makespan > 0:
            gap = (agent_makespan - optimal_makespan) / optimal_makespan
            gap_percent = gap * 100
        else:
            gap = 0.0
            gap_percent = 0.0
        
        # Determine if LLM was successful
        llm_success = (method_used == 'llm')
        
        # Create run record
        record = RunRecord(
            run_id=self._run_counter,
            timestamp=datetime.now().isoformat(),
            instance_name=instance_name,
            agent_makespan=agent_makespan,
            optimal_makespan=optimal_makespan,
            gap=gap,
            gap_percent=gap_percent,
            is_feasible=is_feasible,
            method_used=method_used,
            correction_attempts=correction_attempts,
            llm_success=llm_success,
            negotiation_rounds=negotiation_rounds,
            messages_exchanged=messages_exchanged,
            metadata=metadata or {}
        )
        
        self.runs.append(record)
        return record
    
    def get_feasibility_rate(self) -> float:
        """
        Calculate the percentage of feasible schedules.
        
        ICDAM Table 3: Feasibility Rate (%)
        
        Returns:
            Feasibility rate as percentage (0-100).
        """
        if not self.runs:
            return 0.0
        
        feasible_count = sum(1 for r in self.runs if r.is_feasible)
        return (feasible_count / len(self.runs)) * 100
    
    def get_llm_success_rate(self) -> float:
        """
        Calculate the percentage of runs where LLM produced valid schedule.
        
        ICDAM Table 3: LLM Success Rate (without fallback)
        
        Returns:
            LLM success rate as percentage (0-100).
        """
        if not self.runs:
            return 0.0
        
        llm_success_count = sum(1 for r in self.runs if r.llm_success)
        return (llm_success_count / len(self.runs)) * 100
    
    def get_avg_cost_gap(self) -> float:
        """
        Calculate average cost gap (makespan deviation from optimal).
        
        ICDAM Table 3: Avg. Cost Gap (%)
        
        Returns:
            Average gap as percentage.
        """
        if not self.runs:
            return 0.0
        
        gaps = [r.gap_percent for r in self.runs if r.is_feasible]
        if not gaps:
            return 0.0
        
        return statistics.mean(gaps)
    
    def get_avg_negotiation_rounds(self) -> float:
        """
        Calculate average negotiation rounds per run.
        
        ICDAM Table 3: Avg. Negotiation Rounds
        
        Returns:
            Average number of negotiation rounds.
        """
        if not self.runs:
            return 0.0
        
        rounds = [r.negotiation_rounds for r in self.runs]
        return statistics.mean(rounds)
    
    def get_avg_correction_attempts(self) -> float:
        """
        Calculate average self-correction attempts per run.
        
        ICDAM Table 3: Avg. Self-Correction Attempts
        
        Returns:
            Average number of correction attempts.
        """
        if not self.runs:
            return 0.0
        
        attempts = [r.correction_attempts for r in self.runs]
        return statistics.mean(attempts)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get all summary statistics as a dictionary.
        
        Returns:
            Dictionary with all computed metrics.
        """
        return {
            'experiment_name': self.experiment_name,
            'total_runs': len(self.runs),
            'start_time': self.start_time,
            'end_time': datetime.now().isoformat(),
            
            # Feasibility metrics
            'feasibility_rate': self.get_feasibility_rate(),
            'llm_success_rate': self.get_llm_success_rate(),
            'fallback_rate': 100 - self.get_llm_success_rate(),
            
            # Optimality metrics
            'avg_cost_gap': self.get_avg_cost_gap(),
            'min_gap': min((r.gap_percent for r in self.runs), default=0),
            'max_gap': max((r.gap_percent for r in self.runs), default=0),
            
            # Efficiency metrics
            'avg_negotiation_rounds': self.get_avg_negotiation_rounds(),
            'avg_correction_attempts': self.get_avg_correction_attempts(),
            'avg_messages': statistics.mean([r.messages_exchanged for r in self.runs]) if self.runs else 0,
        }
    
    def print_summary(self) -> None:
        """
        Print a formatted summary report of all metrics.
        
        ICDAM 2025 Table 3: Formatted output for paper/presentation.
        """
        stats = self.get_summary_stats()
        
        # Header
        print("\n")
        print("╔" + "═" * 68 + "╗")
        print("║" + " ICDAM 2025 - SIMULATION METRICS REPORT ".center(68) + "║")
        print("║" + f" Table 3: Performance Evaluation ".center(68) + "║")
        print("╠" + "═" * 68 + "╣")
        
        # Experiment Info
        print("║" + f" Experiment: {stats['experiment_name']:<54}" + "║")
        print("║" + f" Total Runs: {stats['total_runs']:<54}" + "║")
        print("╠" + "═" * 68 + "╣")
        
        # Section 1: Feasibility Metrics
        print("║" + " FEASIBILITY METRICS ".center(68, "─") + "║")
        print("║" + f"   Feasibility Rate:        {stats['feasibility_rate']:>6.1f}%".ljust(67) + "║")
        print("║" + f"   LLM Success Rate:        {stats['llm_success_rate']:>6.1f}%".ljust(67) + "║")
        print("║" + f"   Solver Fallback Rate:    {stats['fallback_rate']:>6.1f}%".ljust(67) + "║")
        print("╠" + "═" * 68 + "╣")
        
        # Section 2: Optimality Metrics (Cost Gap)
        print("║" + " OPTIMALITY METRICS (vs OR-Tools Optimal) ".center(68, "─") + "║")
        print("║" + f"   Avg. Cost Gap:           {stats['avg_cost_gap']:>6.2f}%".ljust(67) + "║")
        print("║" + f"   Min Gap:                 {stats['min_gap']:>6.2f}%".ljust(67) + "║")
        print("║" + f"   Max Gap:                 {stats['max_gap']:>6.2f}%".ljust(67) + "║")
        print("╠" + "═" * 68 + "╣")
        
        # Section 3: Efficiency Metrics
        print("║" + " EFFICIENCY METRICS ".center(68, "─") + "║")
        print("║" + f"   Avg. Negotiation Rounds: {stats['avg_negotiation_rounds']:>6.1f}".ljust(67) + "║")
        print("║" + f"   Avg. Correction Attempts:{stats['avg_correction_attempts']:>6.1f}".ljust(67) + "║")
        print("║" + f"   Avg. Messages Exchanged: {stats['avg_messages']:>6.1f}".ljust(67) + "║")
        print("╠" + "═" * 68 + "╣")
        
        # Interpretation
        print("║" + " INTERPRETATION ".center(68, "─") + "║")
        
        # Feasibility interpretation
        if stats['feasibility_rate'] == 100:
            print("║" + "   ✓ PERFECT: All schedules are feasible (Grounding works!)".ljust(67) + "║")
        elif stats['feasibility_rate'] >= 90:
            print("║" + "   ✓ GOOD: High feasibility rate".ljust(67) + "║")
        else:
            print("║" + "   ✗ WARNING: Low feasibility rate - check constraints".ljust(67) + "║")
        
        # LLM performance interpretation
        if stats['llm_success_rate'] >= 50:
            print("║" + "   ✓ LLM: Produces valid schedules majority of time".ljust(67) + "║")
        else:
            print("║" + "   → LLM: Relies on solver fallback (expected with mock mode)".ljust(67) + "║")
        
        # Gap interpretation
        if stats['avg_cost_gap'] == 0:
            print("║" + "   ✓ OPTIMAL: Agent achieves optimal makespan".ljust(67) + "║")
        elif stats['avg_cost_gap'] <= 5:
            print("║" + "   ✓ NEAR-OPTIMAL: Within 5% of optimal".ljust(67) + "║")
        elif stats['avg_cost_gap'] <= 15:
            print("║" + "   → ACCEPTABLE: Within 15% of optimal".ljust(67) + "║")
        else:
            print("║" + "   ✗ SUBOPTIMAL: Significant gap from optimal".ljust(67) + "║")
        
        # Footer
        print("╚" + "═" * 68 + "╝")
    
    def print_run_details(self) -> None:
        """Print detailed information for each run."""
        if not self.runs:
            print("No runs recorded yet.")
            return
        
        print("\n" + "=" * 70)
        print("DETAILED RUN LOG")
        print("=" * 70)
        
        for run in self.runs:
            print(f"\n[Run {run.run_id}] Instance: {run.instance_name}")
            print(f"  Makespan: {run.agent_makespan} (Optimal: {run.optimal_makespan}, Gap: {run.gap_percent:.1f}%)")
            print(f"  Feasible: {run.is_feasible}, Method: {run.method_used}")
            print(f"  Corrections: {run.correction_attempts}, Negotiations: {run.negotiation_rounds}")
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all data for saving/analysis.
        
        Returns:
            Dictionary with summary stats and all run records.
        """
        return {
            'summary': self.get_summary_stats(),
            'runs': [
                {
                    'run_id': r.run_id,
                    'instance_name': r.instance_name,
                    'agent_makespan': r.agent_makespan,
                    'optimal_makespan': r.optimal_makespan,
                    'gap_percent': r.gap_percent,
                    'is_feasible': r.is_feasible,
                    'method_used': r.method_used,
                    'correction_attempts': r.correction_attempts,
                    'negotiation_rounds': r.negotiation_rounds,
                    'messages_exchanged': r.messages_exchanged,
                    'timestamp': r.timestamp
                }
                for r in self.runs
            ]
        }


# =============================================================================
# Module Test
# =============================================================================

def main():
    """Test the metrics module with sample data."""
    print("=" * 70)
    print("SIMULATION METRICS - Module Test")
    print("=" * 70)
    
    # Create metrics collector
    metrics = SimulationMetrics(experiment_name="Test_Experiment")
    
    # Simulate some runs
    test_runs = [
        # (agent_makespan, optimal, feasible, negotiation_rounds, method, attempts)
        (42, 42, True, 2, 'llm', 1),           # Perfect LLM result
        (45, 42, True, 3, 'llm', 2),           # LLM with small gap
        (42, 42, True, 1, 'solver_fallback', 3),  # Fallback to optimal
        (48, 42, True, 2, 'llm', 1),           # LLM with larger gap
        (42, 42, True, 2, 'solver_fallback', 3),  # Another fallback
    ]
    
    print("\n[Recording test runs...]")
    for i, (agent, optimal, feasible, rounds, method, attempts) in enumerate(test_runs):
        record = metrics.record_run(
            agent_makespan=agent,
            optimal_makespan=optimal,
            is_feasible=feasible,
            negotiation_rounds=rounds,
            instance_name=f"test_instance_{i+1}.sm",
            method_used=method,
            correction_attempts=attempts,
            messages_exchanged=rounds * 2
        )
        print(f"  Run {record.run_id}: makespan={agent}, gap={record.gap_percent:.1f}%, method={method}")
    
    # Print summary
    metrics.print_summary()
    
    # Print run details
    metrics.print_run_details()
    
    print("\n" + "=" * 70)
    print("METRICS MODULE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
