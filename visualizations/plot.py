from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import sys


def load_benchmark_data(filename):
    """Load benchmark JSON data from file"""
    with open(filename, "r") as f:
        return json.load(f)


def process_data(json_data):
    """Process benchmark data and calculate statistics"""
    grouped_by_function = defaultdict(
        lambda: {"n_values": [], "times": []}
    )

    # Group data by function_id
    obj = json_data  if isinstance(json_data,list ) else  [json_data]
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
                    # Find existing n and add to it
                    n_idx = grouped_by_function[func_id]["n_values"].index(n)
                    grouped_by_function[func_id]["times"][n_idx].append(time_value)
                else:
                    # New n value
                    grouped_by_function[func_id]["n_values"].append(n)
                    grouped_by_function[func_id]["times"].append([time_value])

    processed_data = {}

    for func_id, func_data in grouped_by_function.items():
        # Sort by n values
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
    """Calculate speedup relative to first function"""
    function_ids = list(processed_data.keys())
    if len(function_ids) == 0:
        return {}

    baseline_func = function_ids[0]
    baseline_data = processed_data[baseline_func]

    speedup_data = {}

    for func_id, func_data in processed_data.items():
        speedup = []
        for i, n in enumerate(func_data["n"]):
            # Find corresponding baseline time
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
    """Plot mean execution times"""
    plt.figure(figsize=(12, 6))

    for func_id, data in processed_data.items():
        plt.plot(data["n"], data["mean"], marker="o", linewidth=2, label=func_id)

    plt.xlabel("Tree Size (n)", fontsize=12)
    plt.ylabel("Mean Execution Time (ms)", fontsize=12)
    plt.title("Mean Execution Time vs Tree Size", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        plt.show()


def plot_speedup(speedup_data, output_file=None):
    """Plot speedup relative to baseline"""
    plt.figure(figsize=(12, 6))

    for func_id, data in speedup_data.items():
        plt.plot(data["n"], data["speedup"], marker="o", linewidth=2, label=func_id)

    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    plt.xlabel("Tree Size (n)", fontsize=12)
    plt.ylabel("Speedup (relative to first method)", fontsize=12)
    plt.title("Speedup vs Tree Size", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_plotter.py <benchmark.json> [--save]")
        print("\nOptions:")
        print("  --save    Save plots to files instead of displaying them")
        sys.exit(1)

    filename = sys.argv[1]
    save_plots = "--save" in sys.argv

    print(f"Loading benchmark data from: {filename}")
    json_data = load_benchmark_data(filename)

    print("Processing data...")
    processed_data = process_data(json_data)
    speedup_data = calculate_speedup(processed_data)

    print(f"Found {len(processed_data)} functions")

    print("\nGenerating plots...")

    gen_plots_dir = Path(__file__).parent / 'gen_plots'
    gen_plots_dir.mkdir(parents=True, exist_ok=True)

    if save_plots:
        plot_mean_times(processed_data, gen_plots_dir / "benchmark_mean.png")
        plot_speedup(speedup_data, gen_plots_dir / "benchmark_speedup.png")
    else:
        plot_mean_times(processed_data)
        plot_speedup(speedup_data)


if __name__ == "__main__":
    main()
