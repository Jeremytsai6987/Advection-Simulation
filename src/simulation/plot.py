import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_binary_file(file_path):
    """
    Load a binary file containing [x, y, value] in float64 format.
    Returns three numpy arrays (x, y, values).
    """
    data = np.fromfile(file_path, dtype=np.float64).reshape(-1, 3)
    return data[:, 0], data[:, 1], data[:, 2]

def reconstruct_global_grid(x, y, values):
    """
    Construct a regular grid from scattered MPI rank data.
    Returns X, Y meshgrid and a correctly ordered Z matrix.
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x or y is empty, check data loading!")

    unique_x = np.sort(np.unique(x))
    unique_y = np.sort(np.unique(y))

    print(f"unique_x shape: {unique_x.shape}, unique_y shape: {unique_y.shape}")

    if len(unique_x) < 2 or len(unique_y) < 2:
        raise ValueError("Not enough unique x or y values to create a meshgrid!")

    X, Y = np.meshgrid(unique_x, unique_y, indexing="ij")
    Z = np.full(X.shape, np.nan)

    x_index = np.searchsorted(unique_x, x)
    y_index = np.searchsorted(unique_y, y)

    Z[x_index, y_index] = values  

    return X, Y, Z

def plot_global_data(X, Y, Z, title, output_path, vmin=0, vmax=1):
    """
    Create a heatmap using pcolormesh to ensure correct alignment.
    """
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, cmap="viridis", shading="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(label="C(x, y)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Global plot saved: {output_path}")

output_dir = "./"
step = 8000  

pattern = os.path.join(output_dir, f"output_rank_*_step_{step}.bin")
rank_files = glob.glob(pattern)

if not rank_files:
    print(f"No files found for step {step} in {output_dir}")
else:
    global_x, global_y, global_values = [], [], []

    for file_path in rank_files:
        print(f"Loading file: {file_path}")
        x, y, values = load_binary_file(file_path)
        global_x.extend(x)
        global_y.extend(y)
        global_values.extend(values)

    global_x = np.array(global_x)
    global_y = np.array(global_y)
    global_values = np.array(global_values)

    X, Y, Z = reconstruct_global_grid(global_x, global_y, global_values)

    plot_global_data(
        X, Y, Z,
        title=f"Simulation Results at Step {step}",
        output_path=os.path.join(output_dir, f"global_plot_step_{step}.png"),
        vmin=0,
        vmax=1
    )
