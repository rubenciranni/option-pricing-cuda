#!/usr/bin/env python3
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time


# Mapping from full function names to short names
NAME_MAPPING = {
    "vanilla_american_binomial_cuda_batch_scheduler_stprcmp_xdovlpunroll_shuffle_trimotm_ds": "without banked",
    "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds": "Adaptive UnRoll Sizing",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_shuffle_trimotm_ds": "Temporal Unrolling",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_ds": "Pruning OTM Nodes",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm": "Double Single Arithmetic", 
    "vanilla_american_binomial_cuda_batch_scheduler_xdovlpunroll_shuffle_trimotm_ds": "Exercise Value Caching",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_vtile_trimotm_ds": "Shuffle Intrinsics",
    "vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds": "None (baseline)"
}


def generate_latex_table(df):
    """Generate a LaTeX table from the DataFrame."""
    # Remove the N Samples column for LaTeX output
    # df_latex = df.drop('N Samples', axis=1)

    
    latex = r"""\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Configuration} & \textbf{Time (ms)} &  \textbf{Effective GNodes/s}   \\
\hline
"""
    
    for _, row in df.iterrows():
        print(row['name'])
        config = row["name"].replace("_", "\\_")
        latex += f"{config} & ${round(row['time'],2)} \\pm {round(row['time_std'],2)}$"
        # latex += f"& {row['Lost Speedup - Mean']} \\pm {row['Lost Speedup - Std']}"
        latex += f"& {round(row['gflops'],2)}\\% "
        latex += f"\\\\\n"
    
    latex += r"""\hline
\end{tabular}
\label{tab:ablation}
\end{table}"""
    
    return latex

if __name__ == "__main__":
    # main()
    # open the csv
    # 1. Load Data
    df=pd.read_json('./data/ablation_study_times.json')
    N = df['n'].iloc[0]
    N_RANDOM_RUNS = df['n_runs'].iloc[0]

    df = df.groupby('function_id')[['time']].agg(['mean', 'std']).reset_index()
    df.columns = ['function_id', 'time', 'time_std']
    df['gflops'] = (N_RANDOM_RUNS* (N*(N+1)/2)/1e9)/(df['time']/1000 )
    df['name'] = df['function_id'].map(NAME_MAPPING)
    df = df.sort_values(by='gflops', ascending=False)
    # ranme the two columns 
    # data = pd.read_csv(output_file)
    print(generate_latex_table(df))




