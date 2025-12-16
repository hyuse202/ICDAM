"""
Benchmark Loop for RCPSP Solver
Evaluates the CP-SAT solver performance on multiple PSPLib instances.
"""

import os
import time
import pandas as pd
from solvers.rcpsp_solver import RCPSPParser, RCPSPSolver


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_instance_files(data_dir, limit=None):
    """
    Get sorted list of .sm files from the data directory.
    
    Args:
        data_dir: Path to the directory containing .sm files
        limit: Maximum number of files to return (None = all files)
        
    Returns:
        list: Sorted list of file paths
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.sm')]
    files.sort()
    
    if limit:
        files = files[:limit]
    
    return [os.path.join(data_dir, f) for f in files]


def benchmark_solver(instance_files, time_limit_per_instance=300):
    """
    Run benchmark on multiple instances.
    
    Args:
        instance_files: List of paths to .sm files
        time_limit_per_instance: Time limit for solver per instance (seconds)
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Processing {len(instance_files)} instances")
    print(f"{'='*70}\n")
    
    for idx, filepath in enumerate(instance_files, 1):
        instance_name = os.path.basename(filepath)
        print(f"[{idx}/{len(instance_files)}] Processing: {instance_name}...", end=" ")
        
        try:
            # Parse the instance
            parser = RCPSPParser(filepath)
            data = parser.parse()
            
            # Solve and measure time
            solver = RCPSPSolver()
            start_time = time.time()
            makespan, status = solver.solve(data, time_limit_seconds=time_limit_per_instance)
            execution_time = time.time() - start_time
            
            # Store results
            results.append({
                'Instance_Name': instance_name,
                'Optimal_Makespan': makespan if makespan is not None else -1,
                'Status': status,
                'Execution_Time_Sec': round(execution_time, 3)
            })
            
            print(f"✓ Makespan={makespan}, Status={status}, Time={execution_time:.3f}s")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append({
                'Instance_Name': instance_name,
                'Optimal_Makespan': -1,
                'Status': 'ERROR',
                'Execution_Time_Sec': 0.0
            })
    
    return pd.DataFrame(results)


def save_results(df, output_path):
    """
    Save results to CSV file.
    
    Args:
        df: Results dataframe
        output_path: Path to output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def print_summary(df):
    """
    Print summary statistics of the benchmark.
    
    Args:
        df: Results dataframe
    """
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")
    
    # Display full results table
    print("Results Table:")
    print(df.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("Statistics:")
    print(f"{'='*70}")
    
    total_instances = len(df)
    optimal_count = len(df[df['Status'] == 'OPTIMAL'])
    feasible_count = len(df[df['Status'] == 'FEASIBLE'])
    error_count = len(df[df['Status'] == 'ERROR'])
    
    print(f"Total Instances:     {total_instances}")
    print(f"Optimal Solutions:   {optimal_count}")
    print(f"Feasible Solutions:  {feasible_count}")
    print(f"Errors:              {error_count}")
    
    # Execution time statistics (excluding errors)
    valid_times = df[df['Status'] != 'ERROR']['Execution_Time_Sec']
    if len(valid_times) > 0:
        print(f"\nExecution Time (seconds):")
        print(f"  Mean:   {valid_times.mean():.3f}")
        print(f"  Median: {valid_times.median():.3f}")
        print(f"  Min:    {valid_times.min():.3f}")
        print(f"  Max:    {valid_times.max():.3f}")
    
    # Makespan statistics (excluding errors)
    valid_makespans = df[df['Optimal_Makespan'] > 0]['Optimal_Makespan']
    if len(valid_makespans) > 0:
        print(f"\nMakespan:")
        print(f"  Mean:   {valid_makespans.mean():.2f}")
        print(f"  Median: {valid_makespans.median():.2f}")
        print(f"  Min:    {valid_makespans.min()}")
        print(f"  Max:    {valid_makespans.max()}")


def main():
    """Main execution function."""
    print("="*70)
    print("RCPSP SOLVER BENCHMARK")
    print("="*70)
    
    # Configuration
    data_dir = os.path.join("data", "raw", "rcpsp", "j30")
    results_dir = "results"
    output_file = os.path.join(results_dir, "solver_baseline_results.csv")
    n_instances = 5  # Test with first 5 instances
    time_limit = 300  # 5 minutes per instance
    
    # Ensure results directory exists
    ensure_directory_exists(results_dir)
    
    # Get instance files
    print(f"\n[1] Scanning directory: {data_dir}")
    instance_files = get_instance_files(data_dir, limit=n_instances)
    print(f"    Found {len(instance_files)} instances to process")
    
    # Run benchmark
    print(f"\n[2] Running benchmark (time limit: {time_limit}s per instance)...")
    results_df = benchmark_solver(instance_files, time_limit_per_instance=time_limit)
    
    # Save results
    print(f"\n[3] Saving results...")
    save_results(results_df, output_file)
    
    # Print summary
    print_summary(results_df)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
