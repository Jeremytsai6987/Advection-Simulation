#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "simulation.h"
#include "nrutil.h"     
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    if (rank == 0 && argc != 8) {
        fprintf(stderr, "Usage: %s <N> <L> <T> <method> <threads> <dim_0> <dim_1>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N, method, num_threads, dim_0, dim_1;
    double L, T, u, v;
    if (rank == 0) {
        N = atoi(argv[1]);
        L = atof(argv[2]);
        T = atof(argv[3]);
        method = atoi(argv[4]);
        num_threads = atoi(argv[5]);
        dim_0 = atoi(argv[6]);
        dim_1 = atoi(argv[7]);


        printf("Simulation parameters:\n");
        printf("  Grid size (N): %d\n", N);
        printf("  Physical length (L): %f\n", L);
        printf("  Total simulation time (T): %f\n", T);
        printf("  Method: %d\n", method);
        printf("  Threads: %d\n", num_threads);
        printf("  Dimensions: %d x %d\n", dim_0, dim_1);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&method, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim_0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim_1, 1, MPI_INT, 0, MPI_COMM_WORLD);


    omp_set_num_threads(num_threads);

    int dims[2] = {dim_0, dim_1}; 
    int coords[2];
    int rows = dims[0];
    int cols = dims[1];
    int neighbors[4] = {-1, -1, -1, -1};
    int Nx, Ny;              
    int global_start_x, global_start_y;
    if (size > 1){
        neighbors[0] = ((rank / cols - 1 + rows) % rows) * cols + rank % cols; // Top neighbor
        neighbors[1] = ((rank / cols + 1) % rows) * cols + rank % cols;       // Bottom neighbor
        neighbors[2] = rank / cols * cols + (rank % cols - 1 + cols) % cols; // Left neighbor
        neighbors[3] = rank / cols * cols + (rank % cols + 1) % cols;       // Right neighbor

        printf("Rank %d: Top=%d, Bottom=%d, Left=%d, Right=%d\n",
            rank, neighbors[0], neighbors[1], neighbors[2], neighbors[3]);

        int rank_row = rank / cols;   
        int rank_col = rank % cols;   

        int Nx_base  = N / rows;
        int Nx_extra = N % rows;   
        int Ny_base  = N / cols;
        int Ny_extra = N % cols;   


        Nx = Nx_base + (rank_row < Nx_extra ? 1 : 0);
        Ny = Ny_base + (rank_col < Ny_extra ? 1 : 0);

        global_start_x = rank_row * Nx_base + (rank_row < Nx_extra ? rank_row : Nx_extra);
        global_start_y = rank_col * Ny_base + (rank_col < Ny_extra ? rank_col : Ny_extra);



    }else {
        neighbors[0] = -1;
        neighbors[1] = -1;
        neighbors[2] = -1;
        neighbors[3] = -1;

        Nx = N;
        Ny = N;
        global_start_x = 0;
        global_start_y = 0;
    }


    void (*solver)(Simulation*);



    Simulation sim;
    sim.rank = rank;
    sim.N = N;

    sim.global_start_x = global_start_x;
    sim.global_start_y = global_start_y;  

    initialize_simulation(&sim, Nx, Ny, global_start_x, global_start_y, L, T, method);

    if (method == 0) {
        solver = lax_update;
    } else if (method == 1) {
        solver = first_order_upwind;
    } else if (method == 2) {
        solver = second_order_upwind;
    }


    char output_dir[] = "simulation"; 
    mkdir(output_dir, 0777); 

    double start_time = MPI_Wtime();
    int test_step = 1;
    for (int step = 0; step < sim.NT; step++) {
        // if (step == 0 || step == sim.NT / 2 || step == sim.NT - 1) {
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     save_to_binary_with_global_coords(&sim, Nx, Ny, global_start_x, global_start_y, step, output_dir);
        //     MPI_Barrier(MPI_COMM_WORLD);

        // }

        if (size > 1) {
            exchange_ghost_cells_mpi(&sim,global_start_x, global_start_y, neighbors);
        } else {
            if (method != 2) {
                update_ghost_cells(&sim);
            } else {
                update_double_ghost_cells(&sim);
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);

        solver(&sim);

        // if (step % 100 == 0) {
        //     save_to_binary_with_global_coords(&sim, Nx, Ny, global_start_x, global_start_y, step, output_dir);
        // }


    }
    double end_time = MPI_Wtime();

    double local_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double total_grid_points = (double)N * (double)N * (double)sim.NT;

        if (max_time <= 0) {
            fprintf(stderr, "Error: max_time is non-positive: %.2f\n", max_time);
        } else {
            double grind_rate = total_grid_points / max_time;
            printf("Simulation completed. Max time: %.2f seconds.\n", max_time);
            printf("Grind rate: %.2f points per second.\n", grind_rate);
        }
    }



    free_simulation(&sim);
    MPI_Finalize();
    return 0;
}








