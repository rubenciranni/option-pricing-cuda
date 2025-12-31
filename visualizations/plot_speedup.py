import sys
import json
from pathlib import Path
from collections import defaultdict

# Plotly imports
import plotly.graph_objects as go
import plotly.colors as pc

#FILE_PATH = './data/thougput.json'
FILE_PATH = './data/baseline-times.json'


def clean_label(label):
    """Remove the specified suffix from function names for cleaner legends"""
    suffix = "vanilla_american_binomial_cuda_"
    return label.replace(suffix, "")

def get_color_palette(n_colors):
    """Generate a custom list of colors based on the number of lines"""
    if n_colors < 1:
        return []
    # 'Turbo' is excellent for high contrast with many lines
    return pc.sample_colorscale('Turbo', [n/(n_colors -1) if n_colors > 1 else 0 for n in range(n_colors)])

def process_data(json_data):
    """Process benchmark data and calculate statistics (Unchanged logic)"""
    grouped_by_function = defaultdict(
        lambda: {"n_values": [], "times": []}
    )

    # Group data by function_id
    obj = json_data if isinstance(json_data, list) else [json_data]
    for benchmark in obj:
        if "runs" not in benchmark:
            continue

        for run in benchmark["runs"]:
            func_id = run["function_id"]

            if (
                "runs" not in run
                or "n" not in run["runs"]
                or "time_ms_mean" not in run["runs"]
            ):
                continue

            for idx, n in enumerate(run["runs"]["n"]):
                time_value = run["runs"]["time_ms_mean"][idx]

                if n in grouped_by_function[func_id]["n_values"]:
                    n_idx = grouped_by_function[func_id]["n_values"].index(n)
                    grouped_by_function[func_id]["times"][n_idx].append(time_value)
                else:
                    grouped_by_function[func_id]["n_values"].append(n)
                    grouped_by_function[func_id]["times"].append([time_value])

    processed_data = {}

    for func_id, func_data in grouped_by_function.items():
        sorted_indices = sorted(range(len(func_data["n_values"])), key=lambda i: func_data["n_values"][i])
        n_values = [func_data["n_values"][i] for i in sorted_indices]

        mean_times = []
        for i in sorted_indices:
            times = func_data["times"][i]
            if times:
                mean_times.append(sum(times) / len(times))
            else:
                mean_times.append(0.0)

        processed_data[func_id] = {
            "n": n_values,
            "mean": mean_times,
        }
    return processed_data

def calculate_speedup(processed_data):
    """Calculate speedup relative to first function (Unchanged logic)"""
    function_ids = list(processed_data.keys())
    if len(function_ids) == 0:
        return {}

    baseline_func = ""
    max_time = 0
    for func_id, func_data in processed_data.items():
        if func_data["mean"][-1] > max_time:
            max_time = func_data["mean"][-1]
            baseline_func = func_id


    baseline_data = processed_data[baseline_func]

    speedup_data = {}

    for func_id, func_data in processed_data.items():
        speedup = []
        for i, n in enumerate(func_data["n"]):
            if n in baseline_data["n"]:
                baseline_idx = baseline_data["n"].index(n)
                baseline_time = baseline_data["mean"][baseline_idx]
                func_time = func_data["mean"][i]
                speedup.append(baseline_time / func_time if func_time > 0 else 0)
            else:
                speedup.append(1.0)

        speedup_data[func_id] = {"n": func_data["n"], "speedup": speedup}
     
    return speedup_data

def plot_mean_times(processed_data, output_file=None):
    """Plot mean execution times using Plotly"""
    
    # 1. Generate Colors
    func_ids = list(processed_data.keys())
    colors = get_color_palette(len(func_ids))
    
    fig = go.Figure()

    # 2. Add Traces
    arr=list(processed_data.items())
    sorted_arr=sorted(arr,key=lambda x: x[1]["mean"][-1])  
    
    for i, (func_id, data) in enumerate(sorted_arr):
        clean_name = clean_label(func_id)
        fig.add_trace(go.Scatter(
            x=data["n"],
            y=data["mean"],
            mode='lines+markers',
            name=clean_name,
            line=dict(color=colors[i], width=2)
        ))

    # 3. Layout Configuration
    fig.update_layout(
        title="Mean Execution Time vs Tree Size",
        xaxis_title="Tree Size (n)",
        yaxis_title="Mean Execution Time (ms)",
        yaxis_type="log", # Log scale
        hovermode="x unified",
        template="plotly_white"
    )

    if output_file:
        fig.write_html(output_file)
        print(f"Saved HTML: {output_file}")
    else:
        fig.show()

def plot_speedup(speedup_data, output_file=None):
    """Plot speedup relative to baseline using Plotly"""
    
    # 1. Generate Colors
    func_ids = list(speedup_data.keys())
    colors = get_color_palette(len(func_ids))

    fig = go.Figure()

    arr = list(speedup_data.items())
    sorted_arr=sorted(arr,key=lambda x: x[1]["speedup"][-1])  
    # 2. Add Traces
    for i, (func_id, data) in enumerate(sorted_arr):
        clean_name = clean_label(func_id)
        fig.add_trace(go.Scatter(
            x=data["n"],
            y=data["speedup"],
            mode='lines+markers',
            name=clean_name,
            line=dict(color=colors[i], width=2)
        ))

    # Add Baseline Line (y=1.0)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Baseline")

    # 3. Layout Configuration
    fig.update_layout(
        title="Speedup vs Tree Size",
        xaxis_title="Tree Size (n)",
        yaxis_title="Speedup (relative to slower method)",
        yaxis_type="log", # Log scale
        hovermode="x unified",
        template="plotly_white"
    )

    if output_file:
        fig.write_html(output_file)
        print(f"Saved HTML: {output_file}")
    else:
        fig.show()

def main():

    print(f"Loading benchmark data from: {filename}")
    json_data =  json.load( open(filename, "r") )

    print("Processing data...")
    processed_data = process_data(json_data)
    speedup_data = calculate_speedup(processed_data)

    print(f"Found {len(processed_data)} functions")

    print("\nGenerating plots...")

    gen_plots_dir = Path(__file__).parent / 'gen_plots'
    gen_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_times(processed_data, gen_plots_dir / "benchmark_mean.html")
    plot_speedup(speedup_data, gen_plots_dir / "benchmark_speedup.html")

if __name__ == "__main__":
    main()