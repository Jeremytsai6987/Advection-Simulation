import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import glob
import os

def load_binary_file(file_path):
    """
    Load a binary file containing [x, y, value] in float64 format.
    Returns three numpy arrays (x, y, values).
    """
    data = np.fromfile(file_path, dtype=np.float64).reshape(-1, 3)
    return data[:, 0], data[:, 1], data[:, 2]

# Directory containing output files
output_dir = "./"  # Adjust as needed
gif_output_path = os.path.join(output_dir, "efficient_simulation_results.gif")

# Find all output files
all_rank_files = sorted(glob.glob(os.path.join(output_dir, "output_rank_*.bin")))

# Group files by steps
steps = {}
for file_path in all_rank_files:
    step_str = file_path.split("_step_")[-1].split(".bin")[0]
    step = int(step_str)
    if step not in steps:
        steps[step] = []
    steps[step].append(file_path)

# Sort steps in ascending order
sorted_steps = sorted(steps.keys())

# Prepare data for animation and calculate global min/max
frames = []
global_min = float("inf")
global_max = float("-inf")

for step in sorted_steps:
    global_x, global_y, global_values = [], [], []
    for file_path in steps[step]:
        x, y, values = load_binary_file(file_path)
        global_x.extend(x)
        global_y.extend(y)
        global_values.extend(values)

    # Update global min and max
    global_min = min(global_min, np.min(global_values))
    global_max = max(global_max, np.max(global_values))

    # Append the frame as numpy arrays
    frames.append((np.array(global_x), np.array(global_y), np.array(global_values)))

# Create the animation
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter([], [], c=[], cmap="viridis", marker="s", s=10, vmin=global_min, vmax=global_max)
ax.set_xlim(-0.5, 0.5)  # Adjust based on your simulation domain
ax.set_ylim(-0.5, 0.5)
ax.set_title("Simulation Results")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
cb = plt.colorbar(sc, ax=ax, label="C(x, y)", ticks=np.linspace(global_min, global_max, 6))

def update(frame):
    """
    Update function for each frame in the animation.
    """
    x, y, values = frame
    sc.set_offsets(np.c_[x, y])
    sc.set_array(values)
    return sc,

ani = FuncAnimation(
    fig, update, frames=frames, blit=True, interval=200  # Adjust interval for speed
)

# Save the animation as a GIF
ani.save(gif_output_path, writer=PillowWriter(fps=20))  # Adjust fps as needed
print(f"Efficient GIF saved: {gif_output_path}")
