from tqdm import tqdm
import math
import subprocess
import json
import plotly.graph_objects as go
import plotly.colors as pc
import kaleido

FILE_DATA_OUTPUT = "../visualizations/data/batch_gnodes_per_second.json"
IMAGE_OUTPUT = "../visualizations/gen_plots/gnodes_per_second_plot.svg"

def run_benchmark(N,N_RANDOM_RUNS,params=[]):
    """Run the benchmark command and return parsed JSON output."""
    # Remove --pty flag for batch execution
    cmd = [
        "srun", "-A", "dphpc", "-t", "60",
        "../build/bin/pricing_cli", "batch-random-benchmark",
        "--n-random-runs", str(N_RANDOM_RUNS),
        "-n", str(N),
        "--output-format", "json"
    ] + params
    
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

def gen_data(print_progress=False):
    N=1024
    data_full = {}
    for _ in tqdm(range(10)):
        for N_RANDOM_RUNS in tqdm( [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]):
            if print_progress:
                print(f"\nRunning benchmark for N={N}, N_RANDOM_RUNS={N_RANDOM_RUNS}...")
            results = run_benchmark(N, N_RANDOM_RUNS)
            for i in results:
                gn_s =(i["n"]*(i["n"]+1)/2 )* N_RANDOM_RUNS/(i["time"]/1000)/1e9 
                if print_progress:
                    print("for fun", i["function_id"], "with n=", i["n"] )
                    print("GN/s:", round(gn_s,2))
                data_full.setdefault(i["function_id"], {}).setdefault(N_RANDOM_RUNS, []).append( gn_s)
                

    summary_data = {}

    # with open(FILE_DATA_OUTPUT, "r") as f:
    #     summary_data=json.load( f)
    for func_id, n_dict in data_full.items():
        summary_data.setdefault(func_id, {"n": [], "mean": [], "std": []})
        for n, gn_s_list in n_dict.items():
            mean_gn_s = sum(gn_s_list) / len(gn_s_list)
            variance = sum((x - mean_gn_s) ** 2 for x in gn_s_list) / len(gn_s_list)
            stddev_gn_s = variance ** 0.5
            
            summary_data[func_id]["n"].append(n)
            summary_data[func_id].setdefault("all_times", []).append(gn_s_list)
            summary_data[func_id]["mean"].append(mean_gn_s)
            summary_data[func_id]["std"].append(stddev_gn_s)

    with open(FILE_DATA_OUTPUT, "w") as f:
        json.dump(summary_data, f, indent=4)
    return summary_data

def get_color_palette(n_colors):
    """Generate a custom list of colors based on the number of lines"""
    if n_colors < 1:
        return []
    # 'Turbo' is excellent for high contrast with many lines
    return pc.sample_colorscale('Turbo', [n/(n_colors -1) if n_colors > 1 else 0 for n in range(n_colors)])


renamed = {
    "vanilla_american_binomial_cuda_batch_nvidia": "cuda_nvidia",
    "vanilla_american_binomial_cuda_batch_naive": "cuda_naive",
    "vanilla_american_binomial_cuda_batch_nvidia_baseline": "cuda_nvidia_baseline",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_ds": "cuda_optimized",
}


def plot_data():
    with open(FILE_DATA_OUTPUT, "r") as f:
        processed_data = json.load(f)
        fig = go.Figure()
        n_functions = len(processed_data)
        colors = get_color_palette(n_functions+3)
        for idx, (func_id, data) in enumerate(processed_data.items()):
            zipped_lists = list(zip(data["n"], data["mean"]))
            data = {
                "n": [x for x, _ in zipped_lists ],
                "mean": [y for x, y in zipped_lists ],
            }
            fig.add_trace(go.Scatter(
                x=data["n"],
                y=data["mean"],
                mode='lines+markers',
                name=renamed.get(func_id, func_id),
                line=dict(color=colors[idx]),
                # error_y=dict(
                #     type='data',
                #     array=data["std"],
                #     visible=True,
                #     color=colors[idx]
                # )
            ))
            # add a x with a label on the max peak

            max_index = data["mean"].index(max(data["mean"]))
            fig.add_annotation(
                x=math.log(data["n"][max_index],10),
                y=math.log(data["mean"][max_index],10),
                text=f"Max: {round(data['mean'][max_index], 2)}",
            )

        # legend position bottom
        fig.update_layout(
            title={
            "text": "GNodes per Second vs Batch Size",
            'xanchor': 'center',
            'y':0.9, # new
            'x':0.5,
            'yanchor': 'top' # new
        },
            xaxis_title="Batch Size",
            yaxis_title="GNodes/s",
            xaxis_type="log",
            yaxis_type="log",
            hovermode="x unified",
            xaxis=dict(
                tickmode='array',
                tickvals=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384],
                ticktext=[str(x) for x in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]]
            ),
            legend=dict(
                x=0,
                y=1,
            ),
            template="plotly_white"
        )

        kaleido.get_chrome_sync()
        fig.write_image(IMAGE_OUTPUT)
        print(f"Saved SVG: {IMAGE_OUTPUT}")



if __name__ == "__main__":
    # gen_data(True)
    plot_data()


