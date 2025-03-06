import numpy as np
import matplotlib.pyplot as plt

# Simulated data for testing
N = 800  # Total grid size
dim_0 = 3  # Number of rows in MPI grid
dim_1 = 3  # Number of columns in MPI grid
Nx = N // dim_0  # Base local rows per rank
Ny = N // dim_1  # Base local columns per rank

# Create an empty global grid
global_grid = np.zeros((N, N))

# Simulate rank output for testing
for rank in range(dim_0 * dim_1):
    row_idx = rank // dim_1
    col_idx = rank % dim_1
    
    start_x = row_idx * Nx
    start_y = col_idx * Ny
    
    local_Nx = Nx + (1 if row_idx < N % dim_0 else 0)
    local_Ny = Ny + (1 if col_idx < N % dim_1 else 0)
    
    # Fill each subdomain with the rank number for testing
    subdomain = np.full((local_Nx, local_Ny), rank + 1)
    global_grid[start_x:start_x+local_Nx, start_y:start_y+local_Ny] = subdomain

# Plot the global grid
plt.figure(figsize=(10, 10))
plt.imshow(global_grid, origin='lower', cmap='viridis')
plt.colorbar(label='Rank Value')
plt.title("Domain Decomposition (Simulated Data)")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("./", dpi=150)

plt.show()
