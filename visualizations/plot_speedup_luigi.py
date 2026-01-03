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
kaleido.get_chrome_sync()


#FILE_PATH = './data/thougput.json'
# FILE_PATH = './data/baseline-times.json'
FILE_PATH = './data/random-benchmark-baselines-with-quantlib-and-nvidia.json'
# FILE_PATH = './data/temp.json'

func_id_transformation = {
    "vanilla_american_binomial_cuda_nvidia": "cuda_nvidia",
    "vanilla_american_binomial_cuda_naive": "cuda_naive",
    "vanilla_american_binomial_cuda_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds": "cuda_optimized",
    "vanilla_american_binomial_cpu_naive": "cpu_naive",
    "vanilla_american_binomial_cpu_quantlib": "cpu_quantlib",
    "vanilla_american_binomial_cpu_trimotm_trimeeon_stprcmp": "cpu_optimized",
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
        title={
            'text' : "Mean Execution Time vs Tree Size",
            'y':0.9, # new
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top' # new
        },
        xaxis_title="Tree Size (n)",
        yaxis_title="Mean Execution Time (ms)",
        yaxis_type="log", # Log scale
        xaxis_type="log", # Log scale
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


DATA_TO_ADD = {
        "benchmark_parameters": {
                "K": 100.0,
                "S": 100.0,
                "T": 0.5,
                "nend": 2500000,
                "nrepetition_at_step": 2,
                "nstart": 65536,
                "nstep": -1,
                "q": 0.015,
                "r": 0.03,
                "sigma": 0.2,
                "type": "Put"
        },
        "runs": [
                {
                        "all_times": [
                                [
                                        15.944805,
                                        15.918976
                                ],
                                [
                                        48.450143,
                                        48.622487
                                ],
                                [
                                        162.923361,
                                        162.903276
                                ],
                                [
                                        587.016705,
                                        587.190346
                                ],
                        ],
                        "do_pass_sanity_check": "true",
                        "function_id": "vanilla_american_binomial_cuda_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds",
                        "hyperparams": [],
                        "id": "vanilla_american_binomial_cuda_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds",
                        "runs": {
                                "n": [
                                        65536,
                                        131072,
                                        262144,
                                        524288,
                                ],
                                "price": [
                                        5.276651,
                                        5.276658,
                                        5.276661,
                                        5.276662,
                                ],
                                "time_ms_mean": [
                                        15.931890500000002,
                                        48.536315,
                                        162.9133185,
                                        587.1035254999999,
                                ],
                                "time_ms_std": [
                                        0.01826386105126725,
                                        0.12186561109681611,
                                        0.014202239700127975,
                                        0.12278272859199567,
                                ]
                        },
                        "sanity_check": []
                },
                    {
                        "all_times": [
                                [
                                        2177.496891,
                                        2177.219694
                                ],
                                [
                                        6830.286181,
                                        6824.628552
                                ],
                                [
                                        23609.614968,
                                        23603.013311
                                ]
                        ],
                        "do_pass_sanity_check": "true",
                        "function_id": "vanilla_american_binomial_cuda_naive",
                        "hyperparams": [],
                        "id": "vanilla_american_binomial_cuda_naive",
                        "runs": {
                                "n": [
                                        65536,
                                        131072,
                                        262144
                                ],
                                "price": [
                                        5.276651,
                                        5.276659,
                                        5.276662
                                ],
                                "time_ms_mean": [
                                        2177.3582925,
                                        6827.457366500001,
                                        23606.3141395
                                ],
                                "time_ms_std": [
                                        0.19600787842448775,
                                        4.000547831337856,
                                        4.6680764317696815
                                ]
                        },
                        "sanity_check": []
                }
                                

        ]
}

def plot_GNodes_per_second(processed_data, output_file=None):
    """Plot GNodes per second using Plotly"""
    
    # 1. Generate Colors
    func_ids = list(processed_data.keys())
    colors = get_color_palette(len(func_ids))
    
    fig = go.Figure()

    # 2. Add Traces
    arr=list(processed_data.items())
    

    sorted_arr=sorted(arr,key=lambda x: x[1]["mean"][-1])  
    n_set = set()
    for i, (func_id, data) in enumerate(sorted_arr):
        if func_id in [ t["function_id"] for  t in DATA_TO_ADD["runs"] ]:
            data_to_add = next( t for t in DATA_TO_ADD["runs"] if t["function_id"] == func_id )
            data['n'].extend( data_to_add["runs"]["n"] )
            data['mean'].extend( data_to_add["runs"]["time_ms_mean"] )
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

        # add a line of heigh 

        

        fig.add_trace(go.Scatter(
            x=ns,
            y=gnodes_per_second,
            mode='lines+markers',
            name=clean_name,
            line=dict(color=colors[i], width=2)
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
    filename = FILE_PATH
    print(f"Loading benchmark data from: {filename}")
    json_data =  json.load( open(filename, "r") )

    print("Processing data...")
    processed_data = process_data(json_data)
    speedup_data = calculate_speedup(processed_data)

    print(f"Found {len(processed_data)} functions")

    print("\nGenerating plots...")

    gen_plots_dir = Path(__file__).parent / 'gen_plots'
    gen_plots_dir.mkdir(parents=True, exist_ok=True)

    # plot_mean_times(processed_data, gen_plots_dir / "benchmark_mean.svg")
    # plot_speedup(speedup_data, gen_plots_dir / "benchmark_speedup.svg")
    plot_GNodes_per_second(processed_data, gen_plots_dir / "benchmark_gnodes_per_second.svg")

if __name__ == "__main__":
    main()