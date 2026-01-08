import pandas as pd
import numpy as np
import os


import math

def median_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n == 0: return None
    
    # 1. Sort the data
    sorted_data = sorted(data)
    
    # 2. Find z-score (approx 1.96 for 95%)
    # Using scipy or hardcoding common values
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    # 3. Calculate Ranks (using the formula from the text)
    low_rank = (n - z * math.sqrt(n)) / 2
    high_rank = 1 + (n + z * math.sqrt(n)) / 2
    
    # Round to integer indices
    # We use floor for lower and ceil for upper to be conservative/safe
    idx_lower = math.floor(low_rank)
    idx_upper = math.ceil(high_rank)
    
    # Clamp indices to be within list bounds (0 to n-1)
    # Note: The formula assumes 1-based indexing, Python is 0-based.
    # So we subtract 1 from the calculated 1-based ranks.
    idx_lower = max(0, idx_lower - 1)
    idx_upper = min(n - 1, idx_upper - 1)

    return sorted_data[len(sorted_data)//2],sorted_data[idx_lower], sorted_data[idx_upper]


output_values = pd.DataFrame(columns=["n"  ,"number_of_dataframe", * [f"{i}_{j}" for i in ["ai_dram", "ai_l2", "ai_l1", "Performance"] for j in ["median", "ci_lower", "ci_upper"]]])

files = os.listdir("./res_ncu_full_test/")
for n in map(lambda x:2**x,range(4, 15)):
    # df = pd.read_csv(f"./res_ncu_full_test/full_test_{n}.csv", thousands=",")
    #check if there are path like ./res_ncu_full_test/full_test_number_{n}.csv where number is a number
    df = pd.DataFrame()
    for file in files:
        if file.startswith("full_test_") and file.endswith(f"_{n}.csv"):
            df_add = pd.read_csv(f"./res_ncu_full_test/{file}", thousands=",")
            df = pd.concat([df, df_add[1:]], ignore_index=True)


    work = (
        df["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"]
        .to_numpy().astype(np.float32) +
        df["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"]
        .to_numpy().astype(np.float32) +
        df["derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2"].to_numpy().astype(np.float32)
    ) * df["smsp__cycles_elapsed.avg.per_second"].to_numpy().astype(np.float32)

    traffic_l2 = df["derived__lts__lts2xbar_bytes.sum.per_second"].to_numpy().astype(np.float32)
    traffic_dram = df["dram__bytes.sum.per_second"].to_numpy().astype(np.float32)
    traffic_l1 = df["derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.per_second"].to_numpy().astype(np.float32)



    arithmeic_intensity_dram = work / traffic_dram
    arithmeic_intensity_l2 = work / traffic_l2  
    arithmeic_intensity_l1 = work / traffic_l1


    median_dram, ci_lower_dram, ci_upper_dram = median_confidence_interval(arithmeic_intensity_dram) 
    median_l2, ci_lower_l2, ci_upper_l2 = median_confidence_interval(arithmeic_intensity_l2) 
    median_l1, ci_lower_l1, ci_upper_l1 = median_confidence_interval(arithmeic_intensity_l1) 
    median_work, ci_lower_work, ci_upper_work = median_confidence_interval(work/1e12)

    output_values = pd.concat(
        [
            output_values,
            pd.DataFrame(
                {
                    "n": [n],
                    "number_of_dataframe": [len(df)],
                }|
                {
                    k : v
                    for k, v in
                    zip(
                        [f"ai_dram_median", "ai_dram_ci_lower", "ai_dram_ci_upper"],
                        [median_dram, ci_lower_dram, ci_upper_dram])
                }|

                {
                    k : v
                    for k, v in
                    zip(
                        [f"ai_l1_median", "ai_l1_ci_lower", "ai_l1_ci_upper"],
                        [median_l1, ci_lower_l1, ci_upper_l1])
                }|{
                    k : v
                    for k, v in
                    zip(
                        [f"ai_l2_median", "ai_l2_ci_lower", "ai_l2_ci_upper"],
                        [median_l2, ci_lower_l2, ci_upper_l2])
                }|
                {
                    k : v
                    for k, v in
                    zip(
                        [f"Performance_median", "Performance_ci_lower", "Performance_ci_upper"],
                        [median_work, ci_lower_work, ci_upper_work])
                }
            ),
        ],
        ignore_index=True,
    )

print(output_values)

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Roofline Lines

L1_points = [
    (0.1, 1_064_672_250_839.87*1e-12),
    (2.00, 21_293_445_016_797.31*1e-12),
    (100, 21_293_445_016_797.31*1e-12)
]

L2_points = [
    (0.1, 217_291_108_622.62*1e-12),
    (9.80, 21_293_445_016_797.31*1e-12),
    (100, 21_293_445_016_797.31*1e-12)
]

DRAM_points = [
    (0.1, 44_101_948_488.24*1e-12),
    (48.28, 21_293_445_016_797.31*1e-12),
    (100, 21_293_445_016_797.31*1e-12)
]


# data 


colors = ["#0072B2", "#E69F00", "#009E73"]


fig = go.Figure()


# Rooflines
fig.add_trace(go.Scatter(
    x=[p[0] for p in L1_points],
    y=[p[1] for p in L1_points],
    mode="lines",
    name="L1 Roofline",
    line=dict(color=colors[0], width=3)
))

fig.add_trace(go.Scatter(
    x=[p[0] for p in L2_points],
    y=[p[1] for p in L2_points],
    mode="lines",
    name="L2 Roofline",
    line=dict(color=colors[1], width=3, dash="dash")
))

fig.add_trace(go.Scatter(
    x=[p[0] for p in DRAM_points],
    y=[p[1] for p in DRAM_points],
    mode="lines",
    name="DRAM Roofline",
    line=dict(color=colors[2], width=3, dash="dot")
))

# Collect all points for connecting lines
all_ai_l1 = []
all_ai_l2 = []
all_ai_dram = []
all_performance = []

for idx, (index, row) in enumerate(output_values.iterrows()):
    n = row["n"]
    ai_l1 = row["ai_l1_median"]
    ai_l2 = row["ai_l2_median"]
    ai_dram = row["ai_dram_median"]
    performance = row["Performance_median"]
    
    # Get confidence intervals
    ai_l1_lower = row["ai_l1_ci_lower"]
    ai_l1_upper = row["ai_l1_ci_upper"]
    ai_l2_lower = row["ai_l2_ci_lower"]
    ai_l2_upper = row["ai_l2_ci_upper"]
    ai_dram_lower = row["ai_dram_ci_lower"]
    ai_dram_upper = row["ai_dram_ci_upper"]
    perf_lower = row["Performance_ci_lower"]
    perf_upper = row["Performance_ci_upper"]
    
    # Collect points for connecting lines
    all_ai_l1.append(ai_l1)
    all_ai_l2.append(ai_l2)
    all_ai_dram.append(ai_dram)
    all_performance.append(performance)
    
    # Show label only for first point
    show_legend = (idx == 0)
    
    # Calculate power of 2
    power = int(np.log2(n))
    show_text = idx == 0 or index == len(output_values) - 1  # Show text every other point

    # L1 point with error bars
    fig.add_trace(go.Scatter(
        x=[ai_l1],
        y=[performance],
        mode="markers",
        marker=dict(color=colors[0], size=8, symbol="circle"),
        name="L1",
        legendgroup="L1",
        showlegend=show_legend,
        error_x=dict(
            type='data',
            symmetric=False,
            array=[ai_l1_upper - ai_l1],
            arrayminus=[ai_l1 - ai_l1_lower],
            color=colors[0]
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=[perf_upper - performance],
            arrayminus=[performance - perf_lower],
            color=colors[0]
        )
    ))
    
 

    # L2 point with error bars
    fig.add_trace(go.Scatter(
        x=[ai_l2],
        y=[performance],
        mode="markers",
        marker=dict(color=colors[1], size=8, symbol="square"),
        name="L2",
        legendgroup="L2",
        showlegend=show_legend,
        error_x=dict(
            type='data',
            symmetric=False,
            array=[ai_l2_upper - ai_l2],
            arrayminus=[ai_l2 - ai_l2_lower],
            color=colors[1]
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=[perf_upper - performance],
            arrayminus=[performance - perf_lower],
            color=colors[1]
        )
    ))
    

    # DRAM point with error bars
    fig.add_trace(go.Scatter(
        x=[ai_dram],
        y=[performance],
        mode="markers",
        marker=dict(color=colors[2], size=8, symbol="diamond"),
        name="DRAM",
        legendgroup="DRAM",
        showlegend=show_legend,
        error_x=dict(
            type='data',
            symmetric=False,
            array=[ai_dram_upper - ai_dram],
            arrayminus=[ai_dram - ai_dram_lower],
            color=colors[2]
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=[perf_upper - performance],
            arrayminus=[performance - perf_lower],
            color=colors[2]
        )
    ))
    
    if show_text:
        for i,t in enumerate([ai_l1, ai_l2, ai_dram]): 
            fig.add_annotation(
                x=math.log(t,10),
                y=math.log(performance,10),
                text=f"2<sup>{power}</sup>",
                showarrow=False,
                xshift=-15,
                yshift=-15,
                font=dict(size=10, color=colors[i]),
                # arrowcolor=colors[i],
                # arrowwidth=1
            )
# Add connecting lines between points of same memory level
fig.add_trace(go.Scatter(
    x=all_ai_l1,
    y=all_performance,
    mode="lines",
    line=dict(color=colors[0], width=1, dash="dot"),
    showlegend=False,
    hoverinfo="skip"
))

fig.add_trace(go.Scatter(
    x=all_ai_l2,
    y=all_performance,
    mode="lines",
    line=dict(color=colors[1], width=1, dash="dot"),
    showlegend=False,
    hoverinfo="skip"
))

fig.add_trace(go.Scatter(
    x=all_ai_dram,
    y=all_performance,
    mode="lines",
    line=dict(color=colors[2], width=1, dash="dot"),
    showlegend=False,
    hoverinfo="skip"
))


fig.update_xaxes(
    type="log",
    title=dict(
        text="$\\text{Arithmetic Intensity (FLOP/Byte)}$",
        font=dict(size=10)
    ),
    tickfont=dict(size=10),
)

fig.update_yaxes(
    type="log",
    range=[np.log10(0.01*10), np.log10(20*10)],
    title=dict(
        text="$\\text{Performance (TFLOP/s)}$",
        font=dict(size=10)
    ),
    tickfont=dict(size=10),
    # tickformat=".2e"
    
)

fig.update_layout(
    legend=dict(
        font=dict(size=10)
    ),
    margin=dict(l=0, r=0, t=0, b=100)
)


fig.update_layout(
    # title={
    #     'text' : "Mean Execution Time vs Tree Size",
    #     'y':0.9, # new
    #     'x':0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top' # new
    # },
    template="plotly_white"
)

fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

fig.show()
fig.write_image("../visualizations/gen_plots/roofline_full_ncu_test.pdf", scale=2)
