#!/bin/bash
#SBATCH --job-name=hybrid_mpi_openmp_2x2_16c  # Job name
#SBATCH --nodes=4
#SBATCH --ntasks=4                           
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00                        
#SBATCH --partition=caslake                    
#SBATCH --output=output/hybrid_2x2_16c_%j.out  # (%j 為 Job ID)
#SBATCH --error=output/hybrid_2x2_16c_%j.err   # (%j 為 Job ID)
#SBATCH --account=mpcs51087                    

export OMP_NUM_THREADS=16

mpirun --bind-to socket ./my_advection 4000 1.0 1.0 0 16 2 2
