from tqdm import tqdm
from collections import defaultdict
import math
import subprocess
from scipy.stats import shapiro
from scipy import stats 
from plotly.subplots import make_subplots
import json
import plotly.graph_objects as go
import plotly.colors as pc
import kaleido

# FILE_DATA_OUTPUT = "../visualizations/data/batch_gnodes_per_second.json"
FILE_DATA_OUTPUT = "../visualizations/data/batch_gnodes_per_second_2.json"
# FILE_DATA_OUTPUT = "../build/temp.json"
IMAGE_OUTPUT = "../visualizations/gen_plots/gnodes_per_second_plot.svg"

# to create the data run the file
#srun --pty -A dphpc -t 60   ./bin/pricing_cli benchmark-random-throughput --output-format json > ../visualizations/data/batch_gnodes_per_second.json


def get_color_palette(n_colors):
    """Generate a custom list of colors based on the number of lines"""
    if n_colors < 1:
        return []
    # 'Turbo' is excellent for high contrast with many lines
    return pc.sample_colorscale('Turbo', [n/(n_colors -1) if n_colors > 1 else 0 for n in range(n_colors)])


renamed = {
    "vanilla_american_binomial_cuda_batch_nvidia": "Podhloznyuk",
    "vanilla_american_binomial_cuda_batch_naive": "Kolb and Pharr",
    "vanilla_american_binomial_cuda_batch_nvidia_baseline": "Podhloznyuk",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_ds": "Our Method",
}

colors = get_color_palette(5)

func_id_to_color = {
    "vanilla_american_binomial_cuda_batch_nvidia": colors[1],
    "vanilla_american_binomial_cuda_batch_naive": colors[2],
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_ds": colors[3],
    # "vanilla_american_binomial_cpu_naive": "cpu_naive",
    "vanilla_american_binomial_cpu_quantlib": colors[4],
    # "vanilla_american_binomial_cpu_trimotm_trimeeon_stprcmp": "cpu_optimized",
    "vanilla_american_binomial_cuda_batch_nvidia_baseline": colors[1],
}

# def get_data():
#     with open(FILE_DATA_OUTPUT, "r") as f:
#         data_to_process = json.load(f)
#         processed_data = defaultdict(lambda: {
#             "mean_latency":{},
#             "all_time":defaultdict(list),
#             "gnodes/s":{},
#         })
#         n = None
#         for tdata in data_to_process: 
#             func_id = tdata["function_id"]
#             batch_size = tdata["n_runs"]
#             time = tdata["time"]
#             processed_data[func_id]["all_time"][batch_size].append(time)
#             if n is None:
#                 n = tdata["n"]

        
#         for func_id, func_data in processed_data.items():
#             for batch_size, times in func_data["all_time"].items():
#                 mean_time = sum(times) / len(times)
#                 median_time = np.median(times)
#                 calulate_gnodes = lambda t: (batch_size * (n*(n+1)/2) / (t / 1000)) / 1e9  # Convert ms to s and nodes to GNodes
#                 gnodes_per_second = calulate_gnodes(mean_time)
#                 gnodes_per_second_median = calulate_gnodes(median_time)
#                 processed_data[func_id]["mean_latency"][batch_size] = mean_time
#                 processed_data[func_id]["gnodes/s"][batch_size] = gnodes_per_second
#                 processed_data[func_id]["median_gnodes/s"][batch_size] = gnodes_per_second_median
#                 q1 = np.percentile(times, 25)
#                 q3 = np.percentile(times, 75)
#                 processed_data[func_id]["latency_iqr"][batch_size] = (q1, q3)

#                 times_sorted = sorted(times)
#                 n = len(times)
#                 z = 1.96  # 95% CI for the median
#                 lower_rank = max(0, int(math.floor((n - z * math.sqrt(n)) / 2)))
#                 upper_rank = min(n - 1, int(math.ceil((n + z * math.sqrt(n)) / 2) - 1))
#                 ci_lower = times_sorted[lower_rank]
#                 ci_upper = times_sorted[upper_rank]
#                 processed_data[func_id]["median_ci"][batch_size] = (ci_lower, ci_upper)

#     return processed_data
        


# def plot_data():
#     processed_data = get_data()
#     fig = go.Figure()
#     n_functions = len(processed_data)
#     colors = get_color_palette(n_functions+3)
#     x_elements = set()
#     total_normal = 0
#     total_tests = 0
#     for idx, (func_id, data) in enumerate(processed_data.items()):
#         batch_sizes = sorted(data["gnodes/s"].keys())
#         gnodes_mean = [data["gnodes/s"][batch_size] for batch_size in batch_sizes]
#         fig.add_trace(go.Scatter(
#             x=batch_sizes,
#             y=gnodes_mean,
#             mode='lines+markers',
#             name=renamed.get(func_id, func_id),
#             line=dict(color=colors[idx]),
#             error_y=dict(
#                 #
#             )
#         ))
#         x_elements.update(batch_sizes)

#         # add violin plot for each point
#         # test shapiro 


#         max_index = gnodes_mean.index(max(gnodes_mean))
#         fig.add_annotation(
#             x=math.log(batch_sizes[max_index],10),
#             y=math.log(gnodes_mean[max_index],10),
#             text=f"Max: {round(gnodes_mean[max_index], 2)}",
#         )
#     # legend position bottom
#     fig.update_layout(
#         xaxis_title="Batch Size",
#         yaxis_title="GNodes/s",
#         xaxis_type="log",
#         yaxis_type="log",
#         hovermode="x unified",
#         xaxis=dict(
#             tickmode='array',
#             tickvals=sorted(x_elements),
#             ticktext=[str(x) for x in sorted(x_elements)]
#         ),
#         legend=dict(
#             x=0,
#             y=1,
#         ),
#         template="plotly_white"
#     )

#     fig.write_image(IMAGE_OUTPUT)
    # print(f"Saved SVG: {IMAGE_OUTPUT}")

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import numpy as np

def get_data():
    with open(FILE_DATA_OUTPUT, "r") as f:
        data_to_process = json.load(f)
        processed_data = defaultdict(lambda: {
            "mean_latency": {},
            "all_time": defaultdict(list),
            "gnodes/s": {},
            "median_gnodes/s": {},
            "latency_iqr": {},
            "median_ci": {},
            "gnodes_ci": {},  # Add CI for gnodes
        })
        n = None
        for tdata in data_to_process: 
            func_id = tdata["function_id"]
            batch_size = tdata["n_runs"]
            time = tdata["time"]
            processed_data[func_id]["all_time"][batch_size].append(time)
            if n is None:
                n = tdata["n"]
        
        for func_id, func_data in processed_data.items():
            for batch_size, times in func_data["all_time"].items():
                mean_time = sum(times) / len(times)
                median_time = np.median(times)
                calculate_gnodes = lambda t: (batch_size * (n*(n+1)/2) / (t / 1000)) / 1e9
                gnodes_per_second = calculate_gnodes(mean_time)
                gnodes_per_second_median = calculate_gnodes(median_time)
                processed_data[func_id]["mean_latency"][batch_size] = mean_time
                processed_data[func_id]["gnodes/s"][batch_size] = gnodes_per_second
                processed_data[func_id]["median_gnodes/s"][batch_size] = gnodes_per_second_median
                
    return processed_data
def plot_data_single():
    processed_data = get_data()
    fig = go.Figure()
    
    colors = get_color_palette(len(processed_data) + 3)
    x_elements = set()
    
    for idx, (func_id, data) in enumerate(processed_data.items()):
        batch_sizes = sorted(data["gnodes/s"].keys())
        gnodes_median = [data["gnodes/s"][batch_size] for batch_size in batch_sizes]
        
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=batch_sizes,
            y=gnodes_median,
            mode='lines+markers',
            name=renamed.get(func_id, func_id),
            line=dict(color= func_id_to_color[func_id], width=2),
            marker=dict(size=8),
            legendgroup=func_id,
        ))
        
        x_elements.update(batch_sizes)
        
        max_index = gnodes_median.index(max(gnodes_median))
        fig.add_annotation(
            x=math.log(batch_sizes[max_index], 10),
            y=math.log(gnodes_median[max_index], 10),
            text=f"{round(gnodes_median[max_index], 2)}",
            showarrow=True,
            arrowhead=2,
        )
    
    fig.update_layout(
        xaxis_title="Batch Size (B)",
        yaxis_title="GigaNodes per Second (GNodes/s)",
        xaxis_type="log",
        yaxis_type="log",
        hovermode="x unified",
        xaxis=dict(
            tickmode='array',
            tickvals=sorted(x_elements),
            ticktext=[str(x) for x in sorted(x_elements)]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            font=dict(size=10)
        ),
        autosize=True,
        yaxis=dict(automargin=True),
        template="plotly_white"
    )
    
    fig.write_image(IMAGE_OUTPUT)
    print(f"Saved image: {IMAGE_OUTPUT}")

if __name__ == "__main__":

    plot_data_single()
    # plot_histogram_data()


