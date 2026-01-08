import json
import math
import plotly.graph_objects as go
import numpy as np


# Read the JSON file
with open('data/surface.json', 'r') as f:
    data = json.load(f)

def three_d_surface_plot(data):
    # Extract data and calculate GNodes/s
    n_values = []
    n_runs_values = []
    gnodes_per_sec = []

    for entry in data:
        n = entry['n']
        n_runs = entry['n_runs']
        time_sec = entry['time']
        
        # Calculate GNodes/s: (n * (n + 1) / 2) / time_sec / 1e9
        gnodes = n_runs*(n * (n + 1) / 2) / (time_sec/1000) / 1e9
        
        n_values.append(n)
        n_runs_values.append(n_runs)
        gnodes_per_sec.append(gnodes)

    # Get unique values for n and n_runs
    unique_n = sorted(list(set(n_values)))
    unique_n_runs = sorted(list(set(n_runs_values)))

    # Create a 2D grid for the surface plot
    Z = np.zeros((len(unique_n_runs), len(unique_n)))

    # Fill the Z matrix with GNodes/s values
    for i, n_run in enumerate(unique_n_runs):
        for j, n in enumerate(unique_n):
            # Find the corresponding GNodes/s value
            for k in range(len(n_values)):
                if n_values[k] == n and n_runs_values[k] == n_run:
                    Z[i, j] = gnodes_per_sec[k]
                    break

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=unique_n,
        y=unique_n_runs,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title='GNodes/s')
    )])

    # Update layout
    fig.update_layout(
        title='3D Surface Plot: GNodes/s vs n and n_runs',
        scene=dict(
            xaxis_title='n',
            yaxis_title='n_runs',
            zaxis_title='GNodes/s',
            xaxis=dict(type='log'),
            yaxis=dict(type='log')
        ),
        width=1000,
        height=800
    )

    # Show the plot
    fig.write_image("prova.svg")
    fig.show()
def heat_map_plot(data):
    """Plot heatmap of GNodes/s vs n and n_runs on log axes with matching numbers."""
    n_values = []
    n_runs_values = []
    gnodes_per_sec = []

    for entry in data:
        n = entry["n"]
        n_runs = entry["n_runs"]
        time_ms = entry["time"]
        gnodes = n_runs * (n * (n + 1) / 2) / (time_ms / 1000) / 1e9

        n_values.append(n)
        n_runs_values.append(n_runs)
        gnodes_per_sec.append(gnodes)

    unique_n = sorted(set(n_values))
    unique_n_runs = sorted(set(n_runs_values))

    Z = np.full((len(unique_n_runs), len(unique_n)), np.nan)
    for i, n_run in enumerate(unique_n_runs):
        for j, n in enumerate(unique_n):
            for k in range(len(n_values)):
                if n_values[k] == n and n_runs_values[k] == n_run:
                    Z[i, j] = gnodes_per_sec[k]
                    break

    fig = go.Figure(
        data=go.Heatmap(
            x=unique_n,
            y=unique_n_runs,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="Nodes Throughput (GNodes/s)"),
            text=np.round(Z, 2),
            texttemplate="%{text}",
            # textfont=dict(color="white"),
            zmin=np.nanmin(Z),
            zmax=np.nanmax(Z),
        )
    )

    fig.update_layout(
        title="Nodes Throughput (GNodes/s) compared to Three Step and Batch Size",

        xaxis=dict(title="Three Step (N)", type="log",
            tickmode='array',
            tickvals=sorted(unique_n),
            ticktext=[f"{n:,}" for n in sorted(unique_n)]
        ),
        yaxis=dict(title="Batch Size (B)", type="log",
            tickmode='array',
            tickvals=sorted(unique_n_runs),
            ticktext=[f"{n:,}" for n in sorted(unique_n_runs)]
        ),
        
        width=1000,
        height=800,
    )

    fig.write_image("prova.svg")
    fig.show()

if __name__ == "__main__":
    # 3d_surface_plot(data)
    heat_map_plot(data)
