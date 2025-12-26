"""
Metrics Calculation Module
ICDAM 2025 - Phase 4: Performance Evaluation
"""

from typing import List, Dict, Any
import numpy as np

def calculate_feasibility_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate the Feasibility Rate: Ratio of FEASIBLE results to total attempts.
    
    Args:
        results: List of result dictionaries, each containing a 'status' key.
    """
    if not results:
        return 0.0
        
    feasible_count = sum(1 for r in results if r.get('status') in ['FEASIBLE', 'OPTIMAL'])
    return feasible_count / len(results)

def calculate_robustness(makespan_static: float, makespan_dynamic: float) -> float:
    """
    Calculate Robustness based on the formula:
    Robustness ≈ 1 / (C_max(Dynamic) - C_max(Static))
    """
    diff = makespan_dynamic - makespan_static
    if diff <= 0:
        return float('inf')
    return 1.0 / diff

def calculate_negotiation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate negotiation efficiency metrics (Table 3).
    """
    total = len(results)
    if total == 0:
        return {}
        
    agreements = sum(1 for r in results if r.get('decision') == 'AGREED')
    total_rounds = sum(r.get('rounds', 1) for r in results)
    total_messages = sum(r.get('messages', 2) for r in results)
    
    return {
        'agreement_rate': agreements / total,
        'avg_consensus_rounds': total_rounds / total,
        'total_communication_cost': total_messages
    }

def calculate_gini_index(values: List[float]) -> float:
    """
    Calculate Gini Index for fairness (0 = perfect equality, 1 = inequality).
    Used for cost/tardiness distribution among agents.
    """
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    if n < 2:
        return 0.0
    
    values = sorted(values)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))

def main():
    """Demonstrate metrics calculation with sample data."""
    print("=" * 70)
    print("ICDAM 2025 - Metrics Calculation Demo (Enhanced)")
    print("=" * 70)
    
    # [1] Feasibility & Efficiency
    sample_results = [
        {'status': 'FEASIBLE'}, {'status': 'FEASIBLE'},
        {'status': 'INFEASIBLE'}, {'status': 'OPTIMAL'}, {'status': 'FEASIBLE'}
    ]
    f_rate = calculate_feasibility_rate(sample_results)
    print(f"\n[1] Feasibility & Efficiency:")
    print(f"    Feasibility Rate: {f_rate:.2f}")
    
    # [2] Negotiation/Coordination
    neg_results = [
        {'decision': 'AGREED', 'rounds': 1, 'messages': 2},
        {'decision': 'AGREED', 'rounds': 2, 'messages': 4},
        {'decision': 'REJECTED', 'rounds': 3, 'messages': 5}
    ]
    neg_metrics = calculate_negotiation_metrics(neg_results)
    print(f"\n[2] Negotiation/Coordination:")
    print(f"    Agreement Rate: {neg_metrics['agreement_rate']:.2f}")
    print(f"    Avg Consensus Rounds: {neg_metrics['avg_consensus_rounds']:.2f}")
    print(f"    Total Communication Cost: {neg_metrics['total_communication_cost']} messages")
    
    # [3] Fairness (Gini Index)
    # Tardiness of 4 different agents/stakeholders
    tardiness = [10.0, 12.0, 11.0, 40.0] 
    gini = calculate_gini_index(tardiness)
    print(f"\n[3] Fairness:")
    print(f"    Gini Index (Tardiness): {gini:.4f} (Lower is fairer)")
    
    # [4] Robustness
    c_max_static = 42.0
    c_max_dynamic = 45.0
    robustness = calculate_robustness(c_max_static, c_max_dynamic)
    print(f"\n[4] Robustness:")
    print(f"    Robustness: {robustness:.4f}")
    
    print("\n" + "=" * 70)
    print("METRICS CALCULATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
