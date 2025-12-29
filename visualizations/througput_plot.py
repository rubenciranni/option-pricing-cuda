import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import plotly.express as px
import plotly.colors as pc


num_lines = 13
my_colors = pc.sample_colorscale('Turbo', [n/(num_lines -1) for n in range(num_lines)])
FILE_PATH = './data/thougput.json'
OUTPUT_PLOT_FILE = lambda x: f'./gen_plots/plot_{x}.html'


def calculate_values(df):
    # Calculates throughput using the formula: (n * n_runs) / time
    df['n_runs'] = pd.to_numeric(df['n_runs'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['n'] = pd.to_numeric(df['n'], errors='coerce')
    
    df.dropna(subset=['n_runs', 'time', 'n'], inplace=True)
    
    total_options = df['n_runs'] 
    
    time_safe = df['time'] / 1000
    df['throughput'] = total_options / time_safe
    df['operation_intensity'] = total_options * (df['n']**2) / time_safe
    print(df)
    
    return df

def operation_intensity_plot(df):
    df_plot = df.copy()
    df_plot.rename(columns={'n_runs': 'n-random-runs'}, inplace=True) 
    
    required_cols = ['n', 'throughput', 'n-random-runs','operation_intensity']
    if not all(col in df_plot.columns for col in required_cols):
        print("Error: DataFrame missing required columns for plotting. Skipping plot.")
        return

    fig = go.Figure()
    
    ns = sorted(df_plot['n'].unique())

    for runs in ns:
        subset = df_plot[df_plot['n'] == runs]
        
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['n-random-runs'],
                y=subset['operation_intensity'],
                mode='lines+markers',
                name=f'{runs}',
                line=dict(width=2),
                marker=dict(size=8)
            ))
    num_lines = len(ns)
    my_colors = pc.sample_colorscale('Turbo', [n/(num_lines -1) for n in range(num_lines)])

    fig.update_layout(
        title=dict(
            text='Operation Intensity vs. Batch Size',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Batch Size',
            type='log',
            tickmode='array',
            tickvals=list(df_plot['n-random-runs'].unique()),
            ticktext=[str(int(x)) for x in sorted(df_plot['n-random-runs'].unique())],
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        yaxis=dict(
            title='Operation Intensity (operations/s)',
            type='log',
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(title='Number of Timesteps'),
        colorway=my_colors
    )

    fig.write_html(OUTPUT_PLOT_FILE('operation_intensity'))

def plot_latency(df):
    df_plot = df.copy()
    df_plot.rename(columns={'n_runs': 'n-random-runs'}, inplace=True) 
    
    required_cols = ['n', 'throughput', 'n-random-runs','time']
    if not all(col in df_plot.columns for col in required_cols):
        print("Error: DataFrame missing required columns for plotting. Skipping plot.")
        return

    fig = go.Figure()
    
    ns = sorted(df_plot['n'].unique())

    for runs in ns:
        subset = df_plot[df_plot['n'] == runs]
        
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['n-random-runs'],
                y=subset['time'],
                mode='lines+markers',
                name=f'{runs}',
                line=dict(width=2),
                marker=dict(size=8)
            ))
    num_lines = len(ns)
    my_colors = pc.sample_colorscale('Turbo', [n/(num_lines -1) for n in range(num_lines)])

    fig.update_layout(
        title=dict(
            text='Latency vs. Batch Size',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Batch Size',
            type='log',
            tickmode='array',
            tickvals=list(df_plot['n-random-runs'].unique()),
            ticktext=[str(int(x)) for x in sorted(df_plot['n-random-runs'].unique())],
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        yaxis=dict(
            title='Latency (s)',
            type='log',
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(title='Number of Timesteps'),
        colorway=my_colors
    )

    fig.write_html(OUTPUT_PLOT_FILE('latency'))

def plot_throughput_inverse_ax(df):
    df_plot = df.copy()
    df_plot.rename(columns={'n_runs': 'n-random-runs'}, inplace=True) 

    
    required_cols = ['n', 'throughput', 'n-random-runs']
    if not all(col in df_plot.columns for col in required_cols):
        print("Error: DataFrame missing required columns for plotting. Skipping plot.")
        return

    fig = go.Figure()
    
    ns = sorted(df_plot['n'].unique())

    for runs in ns:
        subset = df_plot[df_plot['n'] == runs]
        
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['n-random-runs'],
                y=subset['throughput'],
                mode='lines+markers',
                name=f'{runs}',
                line=dict(width=2),
                marker=dict(size=8)
            ))

    fig.update_layout(
        title=dict(
            text='Throughput vs. Batch Size',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Batch Size',
            type='log',
            tickmode='array',
            tickvals=list(df_plot['n-random-runs'].unique()),
            ticktext=[str(int(x)) for x in sorted(df_plot['n-random-runs'].unique())],
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        yaxis=dict(
            title='Throughput (options/s)',
            type='log',
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(title='Number of Timesteps'),
        colorway=my_colors
    )

    fig.write_html(OUTPUT_PLOT_FILE('throughput_inverse_ax'))

def plot_throughput(df):
    df_plot = df.copy()
    df_plot.rename(columns={'n_runs': 'n-random-runs'}, inplace=True) 
    
    required_cols = ['n', 'throughput', 'n-random-runs']
    if not all(col in df_plot.columns for col in required_cols):
        print("Error: DataFrame missing required columns for plotting. Skipping plot.")
        return

    fig = go.Figure()
    
    random_runs = sorted(df_plot['n-random-runs'].unique())

    for runs in random_runs:
        subset = df_plot[df_plot['n-random-runs'] == runs]
        
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['n'],
                y=subset['throughput'],
                mode='lines+markers',
                name=f'{runs}',
                line=dict(width=2),
                marker=dict(size=8)
            ))

    num_lines = len(random_runs)
    my_colors = pc.sample_colorscale('Turbo', [n/(num_lines -1) for n in range(num_lines)])

    fig.update_layout(
        title=dict(
            text='Throughput vs. Number of Timesteps',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Number of Timesteps',
            type='log',
            tickmode='array',
            tickvals=list(df_plot['n'].unique()),
            ticktext=[str(int(x)) for x in sorted(df_plot['n'].unique())],
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        yaxis=dict(
            title='Throughput (options/s)',
            type='log',
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dash'
        ),
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        # color_discrete_sequence=px.colors.qualitative.Alphabet,
        colorway=my_colors,
        legend=dict(title='Batch Size')
    )

    fig.write_html(OUTPUT_PLOT_FILE('throughput'))

def main():
    try:
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)

        df = calculate_values(df)

        plot_throughput(df)

        plot_throughput_inverse_ax(df)

        plot_latency(df)
        
        # operation_intensity_plot(df)

    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{FILE_PATH}'. Check the JSON format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
