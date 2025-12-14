import pandas as pd
import matplotlib.pyplot as plt
import json
import os

FILE_PATH = '/home/gspadaccini/option-pricing-cuda/build/out.json'
OUTPUT_PLOT_FILE = lambda x: f'plot_{x}.png'

def calculate_throughput(df):
    # Calculates throughput using the formula: (n * n_runs) / time
    df['n_runs'] = pd.to_numeric(df['n_runs'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['n'] = pd.to_numeric(df['n'], errors='coerce')
    
    df.dropna(subset=['n_runs', 'time', 'n'], inplace=True)
    
    total_options = df['n_runs']  * df['n']
    
    time_safe = df['time'] / 1000
    df['throughput'] = total_options / time_safe
    print(df)
    
    return df

def plot_latency(df):
    
    df.rename(columns={'n_runs': 'n-random-runs'}, inplace=True) 
    
    required_cols = ['n', 'throughput', 'n-random-runs','time']
    if not all(col in df.columns for col in required_cols):
        print("Error: DataFrame missing required columns for plotting. Skipping plot.")
        return

    plt.figure(figsize=(12, 6))
    
    random_runs = sorted(df['n-random-runs'].unique())

    for runs in random_runs:
        subset = df[df['n-random-runs'] == runs]
        
        if not subset.empty:
            plt.plot(
                subset['n'],
                subset['time'],
                marker='o',
                linestyle='-',
                label=f'$N_{{runs}}$: {runs}'
            )

    # --- Plot Customization ---
    plt.title(f'Latency vs. Input Size (n)', fontsize=16)
    plt.xlabel('Input Size (n)', fontsize=14)
    plt.ylabel('Latency (seconds)', fontsize=14)
    plt.xscale("log") 
    plt.yscale("log")
    
    plt.xticks(list(df['n'].unique()))

    plt.grid(True, which="major", ls="--", alpha=0.7)
    plt.legend(title='Number of Runs', loc='best')
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_FILE('latency'))

def plot_throughput(df):
    
    df.rename(columns={'n_runs': 'n-random-runs'}, inplace=True) 
    
    required_cols = ['n', 'throughput', 'n-random-runs']
    if not all(col in df.columns for col in required_cols):
        print("Error: DataFrame missing required columns for plotting. Skipping plot.")
        return

    plt.figure(figsize=(12, 6))
    
    random_runs = sorted(df['n-random-runs'].unique())

    for runs in random_runs:
        subset = df[df['n-random-runs'] == runs]
        
        if not subset.empty:
            plt.plot(
                subset['n'],
                subset['throughput'],
                marker='o',
                linestyle='-',
                label=f'$N_{{runs}}$: {runs}'
            )

    # --- Plot Customization ---
    plt.title(f'Throughput vs. Input Size (n)', fontsize=16)
    plt.xlabel('Input Size (n)', fontsize=14)
    plt.ylabel('Throughput (Options/s)', fontsize=14)
    
    plt.xticks(list(df['n'].unique()))
    plt.xscale("log") 
    plt.yscale("log")

    plt.grid(True, which="major", ls="--", alpha=0.7)
    plt.legend(title='Number of Runs', loc='best')
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_FILE('throughput'))

def main():
    try:
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)

        df = calculate_throughput(df)

        plot_throughput(df)

        plot_latency(df)

    except FileNotFoundError:
        print(f"Error: The file '{{FILE_PATH}}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{{FILE_PATH}}'. Check the JSON format.")
    except Exception as e:
        print(f"An unexpected error occurred: {{e}}")

if __name__ == '__main__':
    main()