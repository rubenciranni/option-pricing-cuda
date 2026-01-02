#!/usr/bin/env python3
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Configuration
N_RUNS = 10  # Number of times to run the benchmark
N = 20000
N_RANDOM_RUNS = 100

# Mapping from full function names to short names
NAME_MAPPING = {
    "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds": "without scheduler",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_shuffle_trimotm_ds": "without unroll",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_ds": "without trimotm",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm": "without ds",
    "vanilla_american_binomial_cuda_batch_scheduler_xdovlpunroll_shuffle_trimotm_ds": "without bkdstptcmp",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_vtile_trimotm_ds": "without shuffle",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds": "all optimization (baseline)"
}

def run_benchmark():
    """Run the benchmark command and return parsed JSON output."""
    # Remove --pty flag for batch execution
    cmd = [
        "srun", "-A", "dphpc", "-t", "60",
        "../build/bin/pricing_cli", "batch-random-benchmark",
        "--n-random-runs", str(N_RANDOM_RUNS),
        "-n", str(N),
        "--skip-sanity-checks",
        "--output-format", "json"
    ]
    
    # Run the process and wait for completion
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for process to complete and get output
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
    
    # Parse JSON output - look for the JSON array in stdout
    lines = stdout.strip().split('\n')
    
    # Find the JSON content (starts with '[')
    json_lines = []
    in_json = False
    bracket_count = 0
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('['):
            in_json = True
            bracket_count += stripped.count('[') - stripped.count(']')
            json_lines.append(line)
        elif in_json:
            json_lines.append(line)
            bracket_count += line.count('[') - line.count(']')
            if bracket_count == 0:
                break
    
    if not json_lines:
        raise ValueError("No JSON output found in stdout")
    
    json_output = '\n'.join(json_lines)
    
    try:
        return json.loads(json_output)
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON. Output was:")
        print("=" * 80)
        print(json_output[:1000])
        print("=" * 80)
        raise

def main():
    # Collect data from multiple runs
    all_results = {name: [] for name in NAME_MAPPING.values()}
    
    print(f"Running benchmark {N_RUNS} times with n={N}, n_random_runs={N_RANDOM_RUNS}")
    print("=" * 80)
    
    successful_runs = 0
    
    for run_idx in range(N_RUNS):
        start_time = time.time()
        
        # Print progress
        print(f"Run {run_idx + 1}/{N_RUNS}...", end=" ", flush=True)
        
        try:
            results = run_benchmark()
            
            elapsed = time.time() - start_time
            print(f"✓ completed in {elapsed:.1f}s")
            successful_runs += 1
            
            for result in results:
                function_id = result["function_id"]
                if function_id in NAME_MAPPING:
                    short_name = NAME_MAPPING[function_id]
                    time_ms = result["time"]
                    all_results[short_name].append(time_ms)
                    
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed (exit code {e.returncode})")
            if e.stderr:
                print(f"  Error output: {e.stderr[:200]}")
            continue
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse JSON output")
            continue
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"Completed {successful_runs}/{N_RUNS} successful runs")
    print("=" * 80)
    
    # Calculate statistics
    baseline_name = "all optimization (baseline)"
    baseline_times = all_results[baseline_name]
    
    if len(baseline_times) == 0:
        print("Error: No baseline data collected!")
        return
    
    baseline_mean = np.mean(baseline_times)
    
    stats_data = []
    for name, times in all_results.items():
        if len(times) == 0:
            print(f"Warning: No data collected for {name}")
            continue
            
        time_mean = np.mean(times)
        time_std = np.std(times)
        speedup_mean = time_mean / baseline_mean
        
        # Calculate speedup std using error propagation
        speedup_values = [t / baseline_mean for t in times]
        speedup_std = np.std(speedup_values)
        
        stats_data.append({
            "Configuration": name,
            "Time (ms) - Mean": f"{time_mean:.2f}",
            "Time (ms) - Std": f"{time_std:.2f}",
            "Lost Speedup - Mean": f"{speedup_mean:.2f}",
            "Lost Speedup - Std": f"{speedup_std:.2f}",
            "N Samples": len(times)
        })
    
    # Create DataFrame and sort
    df = pd.DataFrame(stats_data)
    
    # Put baseline first, then sort others by speedup
    baseline_row = df[df["Configuration"] == baseline_name]
    other_rows = df[df["Configuration"] != baseline_name].copy()
    
    # Sort by lost speedup (extract numeric value)
    other_rows["_speedup_sort"] = other_rows["Lost Speedup - Mean"].str.replace("", "").astype(float)
    other_rows = other_rows.sort_values("_speedup_sort").drop("_speedup_sort", axis=1)
    
    df = pd.concat([baseline_row, other_rows], ignore_index=True)
    
    # Display results
    print("\nBENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Save to CSV
    output_file = "ablation_study_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    latex_file = "ablation_study_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to {latex_file}")

def generate_latex_table(df):
    """Generate a LaTeX table from the DataFrame."""
    # Remove the N Samples column for LaTeX output
    df_latex = df.drop('N Samples', axis=1)
    
    latex = r"""\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Configuration} & \textbf{Time (ms)} & \textbf{Time Std} & \textbf{Lost Speedup} & \textbf{Speedup Std} \\
\hline
"""
    
    for _, row in df_latex.iterrows():
        config = row["Configuration"].replace("_", "\\_")
        latex += f"{config} & {row['Time (ms) - Mean']} & {row['Time (ms) - Std']} & {row['Lost Speedup - Mean']} & {row['Lost Speedup - Std']} \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\caption{Ablation study results showing mean and standard deviation of execution time and lost speedup when each optimization is removed compared to the baseline with all optimizations (N="""
    latex += f"{successful_runs} runs)."
    latex += r"""
}
\label{tab:ablation}
\end{table}"""
    
    return latex

if __name__ == "__main__":
    main()