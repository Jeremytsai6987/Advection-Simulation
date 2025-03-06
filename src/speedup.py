import matplotlib.pyplot as plt

# Data
problem_sizes = ['800x800', '1600x1600', '3200x3200', '6400x6400', '12800x12800', '25600x25600']
cores = [1, 2, 4, 8, 16, 32]
speedup = [1.0, 1.68, 1.56, 2.25, 4.44, 5.37]  # From your table

# Plot: Speedup vs Cores for the largest problem size
plt.figure(figsize=(8, 6))
plt.plot(cores, speedup, marker='o', linestyle='-', label='25600x25600')
plt.title('Speedup vs Number of Cores for Largest Problem Size')
plt.xlabel('Number of Cores')
plt.ylabel('Parallel Speedup')
plt.xticks(cores)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig('speedup_vs_cores_large_problem.png')
plt.show()