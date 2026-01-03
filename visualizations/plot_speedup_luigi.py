import sys
import math
import json
from pathlib import Path
from collections import defaultdict

from scipy.stats import shapiro
# Plotly imports
import plotly.graph_objects as go
import plotly.colors as pc
import kaleido


#FILE_PATH = './data/thougput.json'
# FILE_PATH = './data/baseline-times.json'
# FILE_PATH = './data/random-benchmark-baselines-with-quantlib-and-nvidia.json'
# FILE_PATH = './data/temp.json'

FILES = [
"vanilla_american_binomial_cpu_quantlib.json",
"vanilla_american_binomial_cuda_naive.json",
"vanilla_american_binomial_cuda_nvidia_baseline.json",
"vanilla_american_binomial_cuda_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds.json"
]

func_id_transformation = {
    "vanilla_american_binomial_cuda_nvidia": "cuda_nvidia",
    "vanilla_american_binomial_cuda_naive": "cuda_naive",
    "vanilla_american_binomial_cuda_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds": "cuda_optimized",
    # "vanilla_american_binomial_cpu_naive": "cpu_naive",
    "vanilla_american_binomial_cpu_quantlib": "cpu_quantlib",
    # "vanilla_american_binomial_cpu_trimotm_trimeeon_stprcmp": "cpu_optimized",
    "vanilla_american_binomial_cuda_nvidia_baseline": "cuda_nvidia_baseline",
}


def clean_label(label):
    """Remove the specified suffix from function names for cleaner legends"""
    
    return func_id_transformation[label]

def get_color_palette(n_colors):
    """Generate a custom list of colors based on the number of lines"""
    if n_colors < 1:
        return []
    # 'Turbo' is excellent for high contrast with many lines
    return pc.sample_colorscale('Turbo', [n/(n_colors -1) if n_colors > 1 else 0 for n in range(n_colors)])

def process_data(json_data):
    """Process benchmark data and calculate statistics (Unchanged logic)"""
    grouped_by_function = defaultdict(
        lambda: {"n_values": [], "times": [], "all_times": []}
    )

    # Group data by function_id
    obj = json_data if isinstance(json_data, list) else [json_data]
    for benchmark in obj:
        if "runs" not in benchmark:
            continue

        for run in benchmark["runs"]:
            func_id = run["function_id"]

            print(run.keys())
            if (
                "runs" not in run
                or "n" not in run["runs"]
                or "all_times" not in run
            ):
                print(f"Skipping invalid run data for function_id: {func_id}")
                continue

            for idx, n in enumerate(run["runs"]["n"]):
                all_times = run["all_times"][idx]
                time_value = sum(all_times) / len(all_times)
                if n in grouped_by_function[func_id]["n_values"]:
                    n_idx = grouped_by_function[func_id]["n_values"].index(n)
                    grouped_by_function[func_id]["times"][n_idx].append(time_value)
                    grouped_by_function[func_id]["all_times"][n_idx].extend(all_times)
                else:
                    grouped_by_function[func_id]["n_values"].append(n)
                    grouped_by_function[func_id]["times"].append([time_value])
                    grouped_by_function[func_id]["all_times"].append(all_times)

    processed_data = {}

    faild_shapiro = 0
    faild_confidence = 0
    total = 0
    for func_id, func_data in grouped_by_function.items():
        sorted_indices = sorted(range(len(func_data["n_values"])), key=lambda i: func_data["n_values"][i])
        n_values = [func_data["n_values"][i] for i in sorted_indices]

        mean_times = []
        all_times = []

        for i in sorted_indices:
            times = func_data["times"][i]
            all_times.append(func_data["all_times"][i])
            
            if times:
                mean = sum(times) / len(times)
                mean_times.append(mean)
                
                pvalue = shapiro(func_data["all_times"][i]).pvalue
                if pvalue > 0.05:
                    faild_shapiro += 1
                std = math.sqrt(sum((i - mean) ** 2 for i in func_data["all_times"][i]) / (len(func_data["all_times"][i]) -1))
                pconfidence = mean + std* 1.96
                nconfidence = mean - std* 1.96
                if not (pconfidence < mean + 0.05 *mean and nconfidence > mean - 0.05 *mean):
                    faild_confidence += 1
                total +=1
            else:
                mean_times.append(0.0)

        processed_data[func_id] = {
            "n": n_values,
            "mean": mean_times,
            "all_times": all_times

        }
    print(f"Shapiro test failed {faild_shapiro} times.")
    print(f"Confidence interval test failed {faild_confidence} times.")
    print(f"Total tests conducted: {total}")
    return processed_data

def calculate_speedup(processed_data):
    """Calculate speedup relative to first function (Unchanged logic)"""
    function_ids = list(processed_data.keys())
    if len(function_ids) == 0:
        return {}

    baseline_func = "vanilla_american_binomial_cuda_nvidia"
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
    """Plot mean execution times using Plotly with violin plots for each point"""
    import numpy as np
    
    # 1. Generate Colors
    func_ids = list(processed_data.keys())
    colors = get_color_palette(len(func_ids))
    
    fig = go.Figure()

    # Collect all y-values to determine the range for tick labels
    all_y_values = []

    # 2. Add Traces
    arr = list(processed_data.items())
    sorted_arr = sorted(arr, key=lambda x: x[1]["mean"][-1])  
    x_axis = set() 
    
    for i, (func_id, data) in enumerate(sorted_arr):
        if func_id not in func_id_transformation:
            continue
        clean_name = clean_label(func_id)
        
        # Log-transform the mean values
        log_mean = [np.log10(val) for val in data["mean"]]
        
        # Add line connecting the means
        fig.add_trace(go.Scatter(
            x=data["n"],
            y=log_mean,
            mode='lines',
            name=clean_name,
            line=dict(color=colors[i], width=2),
            legendgroup=i,
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add violin plot for each point
        for j, (n, mean_time) in enumerate(zip(data["n"], data["mean"])):
            if j < len(data['all_times']):
                all_times = data['all_times'][j]
                # Log-transform the individual times
                log_times = [np.log10(t) for t in all_times if t > 0]  # Filter out zeros/negatives
                all_y_values.extend(all_times)
                
                fig.add_trace(go.Violin(
                    x=[str(n)] * len(log_times),
                    y=log_times,
                    name=clean_name,
                    legendgroup=i,
                    scalegroup=i,
                    side='negative',
                    line_color=colors[i],
                    meanline_visible=True,
                    points=False,
                    showlegend=False,
                    opacity=0.6,
                    width=0.7
                ))
        
        x_axis.update(data["n"])

    # Create nice tick values for the y-axis
    y_min, y_max = min(all_y_values), max(all_y_values)
    log_min, log_max = np.floor(np.log10(y_min)), np.ceil(np.log10(y_max))
    tick_vals = [10**i for i in range(int(log_min), int(log_max)+1)]
    tick_vals_log = [np.log10(v) for v in tick_vals]

    # 3. Layout Configuration
    fig.update_layout(
        xaxis_title="Time Steps (n)",
        yaxis_title="Mean Execution Time (ms)",
        yaxis_type="linear",  # Changed to linear since we're plotting log-transformed data
        yaxis=dict(
            tickmode='array',
            tickvals=tick_vals_log,
            ticktext=[str(int(v)) if v >= 1 else f"{v:.1g}" for v in tick_vals]
        ),
        xaxis_type="log",
        xaxis=dict(
            tickvals=sorted(x_axis),
            ticktext=[f"{val:,}" for val in sorted(x_axis)]
        ),
        hovermode="x unified",
        template="plotly_white"
    )

    if output_file:
        fig.write_image(output_file)
        print(f"Saved SVG: {output_file}")
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
        title={
            'text' : "Speedup vs Tree Size",
            'y':0.9, # new
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top' # new
        },
        xaxis_title="Tree Size (n)",
        yaxis_title="Speedup (relative to slower method)",
        yaxis_type="log", # Log scale
        xaxis_type="log",
        hovermode="x unified",
        template="plotly_white"
    )

    if output_file:
        fig.write_image(output_file)
        print(f"Saved SVG: {output_file}")
    else:
        fig.show()



def plot_GNodes_per_second(processed_data, output_file=None):
    """Plot GNodes per second using Plotly with violin plots"""
    
    # 1. Generate Colors
    func_ids = list(processed_data.keys())
    colors = get_color_palette(len(func_ids))
    
    fig = go.Figure()

    # 2. Add Traces
    arr=list(processed_data.items())
    

    sorted_arr=sorted(arr,key=lambda x: x[1]["mean"][-1])  
    n_set = set()
    for i, (func_id, data) in enumerate(sorted_arr):
        clean_name = clean_label(func_id)
        ns,gnodes_per_second = zip(*[
           (n, (n * (n + 1) / 2) / (time_ms / 1000) / 1e9 if time_ms > 0 else 0)
            for n, time_ms in zip(data["n"], data["mean"])
            if n > 100
        ])
        # make shapiro test
        
        ns= list(ns)
        gnodes_per_second=list(gnodes_per_second)
        max_index = gnodes_per_second.index(max(gnodes_per_second))
        # Determine arrow direction based on gnodes value
        ay_offset = 20 if gnodes_per_second[max_index] < 0.10 else -20    
        print(ay_offset,math.log(gnodes_per_second[max_index]))

        fig.add_annotation(
            x=math.log(ns[max_index],10),
            y=math.log(gnodes_per_second[max_index],10),
            text=f"Max: {round(gnodes_per_second[max_index], 2)}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black",
            ay=ay_offset,
            font=dict(size=10, color="black")
        )
        n_set.update(set(ns))

        # Add main line trace
        fig.add_trace(go.Scatter(
            x=ns,
            y=gnodes_per_second,
            mode='lines+markers',
            name=clean_name,
            line=dict(color=colors[i], width=2),
            legendgroup=i,
            showlegend=True
        ))
        
        # Add violin plots for each point
        for j, (n, gnodes) in enumerate(zip(ns, gnodes_per_second)):
            # Get all times for this data point
            idx_in_original = data["n"].index(n) if n in data["n"] else -1
            if idx_in_original >= 0 and idx_in_original < len(data['all_times']):
                all_times = data['all_times'][idx_in_original]
                # Convert times to gnodes
                all_gnodes = [(n * (n + 1) / 2) / (t / 1000) / 1e9 if t > 0 else 0 for t in all_times]
                
                fig.add_trace(go.Violin(
                    x=[str(n)] * len(all_gnodes),
                    y=all_gnodes,
                    name=f"{clean_name} (n={n})",
                    legendgroup=i,
                    scalegroup=i,
                    side='negative',
                    line_color=colors[i],
                    meanline_visible=True,
                    points=False,
                    showlegend=False,
                    opacity=0.6
                ))
    # 3. Layout Configuration
    fig.update_layout(
        title={
            'text' : "Effective GNodes/s vs Tree Size",
            'y':0.9, # new
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top' # new
        },
        xaxis_title="Tree Size (n)",
        yaxis_title="Effective GNodes/s",
        yaxis_type="log", # Log scale
        xaxis_type="log", # Log scale, make it power of 2
        xaxis=dict(
            tickmode='array',
            tickvals=sorted(n_set),
            ticktext=[f"{n}" for n in sorted(n_set)]
        ),
        hovermode="x unified",
        template="plotly_white",
        #legend on the bottom
        legend=dict(
          
                x=0,
                y=1,
        ),


    )
    if output_file:
        fig.write_image(output_file)
        print(f"Saved SVG: {output_file}")
        fig.write_html(output_file.replace(".svg", ".html"))
        print(f"Saved HTML: {output_file}")
    else:
        fig.show()

def main():
    json_data ={ "runs": [] }
    for file in FILES:
        filename = Path(__file__).parent / 'data' / file
        print(f"Loading benchmark data from: {filename}")
        with open(filename, 'r') as f:
            json_data_t = json.load(f)
        json_data['runs'].extend( json_data_t['runs'] )

    print("Processing data...")
    processed_data = process_data(json_data)
    speedup_data = calculate_speedup(processed_data)

    print(f"Found {len(processed_data)} functions")

    print("\nGenerating plots...")

    gen_plots_dir = Path(__file__).parent / 'gen_plots'
    gen_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_times(processed_data, gen_plots_dir / "benchmark_mean.svg")
    # plot_GNodes_per_second(processed_data, gen_plots_dir / "benchmark_gnodes_per_second.svg")
    # plot_speedup(speedup_data, gen_plots_dir / "benchmark_speedup.svg")

if __name__ == "__main__":
    main()