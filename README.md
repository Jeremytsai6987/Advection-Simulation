
---

# Simulation Program for Advection Equation

This repository contains a program to simulate the advection equation using MPI and OpenMP for parallel computation. The program is designed for use on clusters with OpenMPI and supports flexible configuration of grid size, physical domain, and computation methods.

---

## Prerequisites

1. **Required Modules**: 
   - Ensure the `openmpi` module is available. Load it before compiling or running the program.

2. **Make Utility**:
   - Make sure you have `make` installed on your system.

---

## Steps to Compile and Run

### 1. Load the Required Module

Before compiling or running, load the `openmpi` module:
```bash
module load openmpi
```

### 2. Compile the Program

Use `make` to compile the program:
```bash
make
```

This will generate the executable file `my_advection`.

To clean up generated files:
```bash
make clean
```

---

### 3. Run the Program

#### Direct Execution
You can directly run the program using `mpirun`:
```bash
mpirun --bind-to core ./my_advection <N> <L> <T> <method> <threads> <dim_0> <dim_1>
```

#### Run via SLURM
To run the program on a cluster using SLURM, create a batch script (e.g., `submit.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=hybrid_3n3r32t  
#SBATCH --nodes=9                           
#SBATCH --ntasks=9                           
#SBATCH --ntasks-per-node=1                  
#SBATCH --cpus-per-task=16                    
#SBATCH --time=00:01:00                     
#SBATCH --partition=caslake                
#SBATCH --output=output/hybrid_3n3r16t_%j.out  
#SBATCH --error=output/hybrid_3n3r16t_%j.err   
#SBATCH --account=mpcs51087                   

# Set the number of OpenMP threads per MPI rank
export OMP_NUM_THREADS=16

# Run the program with 4 MPI ranks (1 per node)
mpirun --bind-to socket ./my_advection 4000 1.0 1.0 0 16 3 3

```

Submit the job using:
```bash
sbatch submit.slurm
```

---

## Input Parameters

| Parameter   | Description                                         | Example Value |
|-------------|-----------------------------------------------------|---------------|
| `<N>`       | Grid size (number of cells per side)                | `4000`        |
| `<L>`       | Physical length of the domain                      | `1.0`         |
| `<T>`       | Total simulation time                              | `1.0`         |
| `<method>`  | Numerical method:                                   |               |
|             | `0` for Lax, `1` for First-Order Upwind, `2` for Second-Order Upwind | `0` |
| `<threads>` | Number of OpenMP threads per MPI process            | `1`           |
| `<dim_0>`   | Number of MPI processes along the x-direction       | `2`           |
| `<dim_1>`   | Number of MPI processes along the y-direction       | `2`           |

---

### Explanation of `dim_0` and `dim_1`

The parameters `dim_0` and `dim_1` determine how the computational grid is divided among the MPI processes:

1. **Total Grid Size (`N`)**:
   - The grid size (`N x N`) is evenly divided into smaller sub-grids.
   - Each sub-grid is handled by an individual MPI process.

2. **Division Along Dimensions**:
   - `dim_0`: Specifies the number of sub-grids (MPI processes) along the x-axis.
   - `dim_1`: Specifies the number of sub-grids (MPI processes) along the y-axis.

   For example:
   - If `N = 4000`, `dim_0 = 2`, and `dim_1 = 2`, the grid will be divided into 4 sub-grids of size `2000 x 2000` each.

3. **MPI Process Grid**:
   - The product `dim_0 * dim_1` must equal the total number of MPI ranks specified.
   - Each rank is assigned a sub-grid for computation.

---

### Example Configurations

#### 1. **Serial Execution**:
Run the program without parallelism:
```bash
./my_advection 4000 1.0 1.0 0 1 1 1
```

#### 2. **MPI Only**:
Run with 2 nodes, 1 process per node:
```bash
mpirun --bind-to core ./my_advection 4000 1.0 1.0 0 1 2 1
```

#### 3. **Hybrid MPI+OpenMP**:
Run with 1 MPI rank per node and 16 threads per rank:
```bash
mpirun --bind-to socket ./my_advection 4000 1.0 1.0 0 16 2 2
```

---

## Output

- The program will output the maximum runtime for all processes and the "grind rate" (grid points processed per second).
- Intermediate results can be written as binary files for visualization or further analysis.

---

## Visualization

![Simulation](/src/simulation/efficient_simulation_results.gif)


---



