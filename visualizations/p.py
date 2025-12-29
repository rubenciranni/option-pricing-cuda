import pandas as pd
import plotly.express as px
import json
import os

# 1. The JSON Data
data_json = """
[
    {
        "do_pass_sanity_check": "true",
        "function_id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds",
        "hyperparams": ["HD102432", "1024", "16", "5", "0", "0"],
        "id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds_@HD102432_1024_16_5_0_0",
        "n": 50000,
        "n_runs": 100,
        "time": 989.927553
    },
    {
        "do_pass_sanity_check": "true",
        "function_id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds",
        "hyperparams": ["HD12832", "128", "16", "5", "0", "0"],
        "id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds_@HD12832_128_16_5_0_0",
        "n": 50000,
        "n_runs": 100,
        "time": 728.753239
    },
    {
        "do_pass_sanity_check": "true",
        "function_id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds",
        "hyperparams": ["HD25632", "256", "16", "5", "0", "0"],
        "id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds_@HD25632_256_16_5_0_0",
        "n": 50000,
        "n_runs": 100,
        "time": 706.179159
    },
    {
        "do_pass_sanity_check": "true",
        "function_id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds",
        "hyperparams": ["HD51232", "512", "16", "5", "0", "0"],
        "id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds_@HD51232_512_16_5_0_0",
        "n": 50000,
        "n_runs": 100,
        "time": 798.491683
    },
    {
        "do_pass_sanity_check": "true",
        "function_id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds",
        "hyperparams": ["HD6432", "64", "16", "5", "0", "0"],
        "id": "vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds_@HD6432_64_16_5_0_0",
        "n": 50000,
        "n_runs": 100,
        "time": 845.701049
    }
]
"""

# 2. Process Data
data = json.loads(data_json)
df = pd.DataFrame(data)

# Extract hyperparameter[1] and rename it to 'THREAD_PER_BLOCK'
df['THREAD_PER_BLOCK'] = df['hyperparams'].apply(lambda x: int(x[1]))
df['str_THREAD_PER_BLOCK'] = df['hyperparams'].apply(lambda x: x[1])

# Create a text label for the bar (Latency + Value)
df['label'] = df.apply(lambda row: f"{row['time']:.2f} ms", axis=1)

# Sort by time (latency) ascending to match the 'waterfall' look of your example
df = df.sort_values('THREAD_PER_BLOCK')

# 3. Create Plot
fig = px.bar(
    df,
    x='THREAD_PER_BLOCK',
    y='time',
    text='label',
    title='Latency vs. Threads per Block',
    labels={'time': 'Latency (ms)'},
    color='str_THREAD_PER_BLOCK',  # Gives each bar a unique color
)

# 4. Style the Layout (Clean look)
fig.update_traces(
    textposition='auto', 
    showlegend=False  # Hide legend since x-axis labels are sufficient
)

fig.update_layout(
    xaxis_title="Threads per Block",
    yaxis_title="Latency (ms)",
    xaxis={'type': 'category'}, # Ensure it treats 64, 128 etc as distinct categories, not a continuous number line
    uniformtext_minsize=8, 
    uniformtext_mode='hide',
    margin=dict(t=50, l=25, r=25, b=25)
    
)

# 5. Save Output
# Save as HTML (Interactive)
fig.write_html("latency_plot.html")