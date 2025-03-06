#include "simulation.h"
#include "nrutil.h" 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>

void initialize_simulation(Simulation *sim, int Nx, int Ny, 
                           int global_start_x, int global_start_y, 
                           double L, double T, int method)
{
    if (method == 2) {
        sim->ghost_layers = 2;  
    } else {
        sim->ghost_layers = 1;  
    }

    sim->L  = L;
    sim->dx = L / (sim->N - 1);  
    sim->Nx = Nx;
    sim->Ny = Ny;

    sim->dt = 1.25e-4;                 
    sim->NT = (int)(T / sim->dt) + 1;

    int gl = sim->ghost_layers; 

    if (gl == 1) {
        sim->Nx_alloc = Nx + 1;
        sim->Ny_alloc = Ny + 1;

    } else if (gl == 2) {
        sim->Nx_alloc = Nx + 3;
        sim->Ny_alloc = Ny + 3;
    } else {
        fprintf(stderr, "Error: Unsupported ghost layer count %d\n", gl);
        exit(1);
    }
    sim->C = dmatrix(0, sim->Nx_alloc, 0, sim->Ny_alloc);
    sim->C_new = dmatrix(0, sim->Nx_alloc, 0, sim->Ny_alloc);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < sim->Nx_alloc; i++) {
        for (int j = 0; j < sim->Ny_alloc; j++) {
            sim->C[i][j] = 0.0;
        }
    }


    #pragma omp parallel for schedule(dynamic) default(none) shared(sim, Nx, Ny, global_start_x, global_start_y, gl, L)
    for (int i = gl; i <= Nx + gl - 1; i++) {
        for (int j = gl; j <= Ny + gl - 1; j++) {
            double global_x = -L / 2.0 + (global_start_x + i) * sim->dx;
            double global_y = -L / 2.0 + (global_start_y + j) * sim->dx;

            if (global_x >= -0.5 && global_x <= 0.5 &&
                global_y >= -0.1 && global_y <= 0.1) {
                sim->C[i][j] = 1.0;
            } else {
                sim->C[i][j] = 0.0;
            }
        }
    }

}
void exchange_ghost_cells_mpi(Simulation *sim, 
                              int global_start_x, 
                              int global_start_y, 
                              int neighbors[4])
{
    MPI_Status status;
    int Nx = sim->Nx;  
    int Ny = sim->Ny;  
    int gl = sim->ghost_layers;  

    double *send_top    = (double*)malloc((Ny + 2 * gl) * sizeof(double)); 
    double *send_bottom = (double*)malloc((Ny + 2 * gl) * sizeof(double)); 
    double *recv_top    = (double*)malloc((Ny + 2 * gl) * sizeof(double));
    double *recv_bottom = (double*)malloc((Ny + 2 * gl) * sizeof(double));

    double *send_left   = (double*)malloc((Nx + 2 * gl) * sizeof(double)); 
    double *send_right  = (double*)malloc((Nx + 2 * gl) * sizeof(double)); 
    double *recv_left   = (double*)malloc((Nx + 2 * gl) * sizeof(double));
    double *recv_right  = (double*)malloc((Nx + 2 * gl) * sizeof(double));

    for (int j = 0; j <= Ny + 2 * gl - 1; j++) {
        for (int layer = 0; layer < gl; layer++) {
            send_top[j + layer * (Ny + 2 * gl)]    = sim->C[1 + layer][j];
            send_bottom[j + layer * (Ny + 2 * gl)] = sim->C[Nx - layer][j];
        }
    }

    for (int i = 0; i <= Nx + 2 * gl - 1; i++) {
        for (int layer = 0; layer < gl; layer++) {
            send_left[i + layer * (Nx + 2 * gl)]  = sim->C[i][1 + layer];
            send_right[i + layer * (Nx + 2 * gl)] = sim->C[i][Ny - layer];
        }
    }

    MPI_Sendrecv(send_top,    (Ny + 2 * gl) * gl, MPI_DOUBLE, neighbors[0], 0,
                 recv_bottom, (Ny + 2 * gl) * gl, MPI_DOUBLE, neighbors[1], 0,
                 MPI_COMM_WORLD, &status);

    MPI_Sendrecv(send_bottom, (Ny + 2 * gl) * gl, MPI_DOUBLE, neighbors[1], 1,
                 recv_top,    (Ny + 2 * gl) * gl, MPI_DOUBLE, neighbors[0], 1,
                 MPI_COMM_WORLD, &status);

    MPI_Sendrecv(send_left,   (Nx + 2 * gl) * gl, MPI_DOUBLE, neighbors[2], 2,
                 recv_right,  (Nx + 2 * gl) * gl, MPI_DOUBLE, neighbors[3], 2,
                 MPI_COMM_WORLD, &status);

    MPI_Sendrecv(send_right,  (Nx + 2 * gl) * gl, MPI_DOUBLE, neighbors[3], 3,
                 recv_left,   (Nx + 2 * gl) * gl, MPI_DOUBLE, neighbors[2], 3,
                 MPI_COMM_WORLD, &status);

    for (int j = 0; j <= Ny + 2 * gl - 1; j++) {
        for (int layer = 0; layer < gl; layer++) {
            sim->C[1 - gl + layer][j]    = recv_top[j + layer * (Ny + 2 * gl)];
            sim->C[Nx + gl - layer][j] = recv_bottom[j + layer * (Ny + 2 * gl)];
        }
    }

    for (int i = 0; i <= Nx + 2 * gl - 1; i++) {
        for (int layer = 0; layer < gl; layer++) {
            sim->C[i][1 - gl + layer]    = recv_left[i + layer * (Nx + 2 * gl)];
            sim->C[i][Ny + gl - layer] = recv_right[i + layer * (Nx + 2 * gl)];
        }
    }

    free(send_top);    free(send_bottom);
    free(recv_top);    free(recv_bottom);
    free(send_left);   free(send_right);
    free(recv_left);   free(recv_right);
}

void update_double_ghost_cells(Simulation *sim) {
    int N = sim->N;

    for (int i = 2; i <= N + 1; i++) {
        sim->C[i][0]       = sim->C[i][N];
        sim->C[i][N + 1]   = sim->C[i][1];
    }

    for (int j = 2; j <= N + 1; j++) {
        sim->C[0][j]       = sim->C[N][j];
        sim->C[N + 1][j]   = sim->C[1][j];
    }

    sim->C[0][0]             = sim->C[N][N];
    sim->C[0][N + 1]         = sim->C[N][1];
    sim->C[N + 1][0]         = sim->C[1][N];
    sim->C[N + 1][N + 1]     = sim->C[1][1];

    for (int i = 1; i <= N + 2; i++) {
        sim->C[i][N + 2] = sim->C[i][2]; 
        sim->C[i][1]     = sim->C[i][N - 1]; 
    }

    for (int j = 1; j <= N + 2; j++) {
        sim->C[N + 2][j] = sim->C[2][j]; 
        sim->C[1][j]     = sim->C[N - 1][j]; 
    }

    sim->C[N + 2][N + 2]     = sim->C[2][2];
    sim->C[N + 2][1]         = sim->C[2][N - 1];
    sim->C[1][N + 2]         = sim->C[N - 1][2];
    sim->C[1][1]             = sim->C[N - 1][N - 1];
}




void update_ghost_cells(Simulation *sim) {
    int N = sim->N;

    for (int i = 1; i <= N; i++) {
        sim->C[i][0]       = sim->C[i][N];
        sim->C[i][N + 1]   = sim->C[i][1];
    }

    for (int j = 1; j <= N; j++) {
        sim->C[0][j]       = sim->C[N][j];
        sim->C[N + 1][j]   = sim->C[1][j];
    }

    sim->C[0][0]             = sim->C[N][N];
    sim->C[0][N + 1]         = sim->C[N][1];
    sim->C[N + 1][0]         = sim->C[1][N];
    sim->C[N + 1][N + 1]     = sim->C[1][1];
}

void second_order_upwind(Simulation *sim) {
    int Nx = sim->Nx;  
    int Ny = sim->Ny;  
    double dx = sim->dx;
    double dt = sim->dt;


    #pragma omp parallel for schedule(dynamic) default(none) shared(sim, Nx, Ny, dx, dt)
    for (int i = 2; i <= Nx + 1; i++) {
        for (int j = 2; j <= Ny + 1; j++) {
            double x = -sim->L / 2.0 + (sim->global_start_x + i - 1) * dx;
            double y = -sim->L / 2.0 + (sim->global_start_y + j - 1) * dx;
            double u = sqrt(2.0) * y;
            double v = -sqrt(2.0) * x;

            double c_ij = sim->C[i][j];
            double c_imj = sim->C[i - 1][j];
            double c_ipj = sim->C[i + 1][j];
            double c_ijm = sim->C[i][j - 1];
            double c_ijp = sim->C[i][j + 1];
            double c_im2j = sim->C[i - 2][j];
            double c_ip2j = sim->C[i + 2][j];
            double c_ijm2 = sim->C[i][j - 2];
            double c_ijp2 = sim->C[i][j + 2];

            double advection_term = 0.0;
            if (u > 0 && v > 0) {
                advection_term = dt * ((u * (4 * c_imj - 3 * c_ij - c_im2j) / (2 * dx)) +
                                       (v * (4 * c_ijm - 3 * c_ij - c_ijm2) / (2 * dx)));
            } else if (u > 0 && v < 0) {
                advection_term = dt * ((u * (4 * c_imj - 3 * c_ij - c_im2j) / (2 * dx)) +
                                       (v * (4 * c_ijp - 3 * c_ij - c_ijp2) / (2 * dx)));
            } else if (u < 0 && v > 0) {
                advection_term = dt * ((u * (4 * c_ipj - 3 * c_ij - c_ip2j) / (2 * dx)) +
                                       (v * (4 * c_ijm - 3 * c_ij - c_ijm2) / (2 * dx)));
            } else if (u < 0 && v < 0) {
                advection_term = dt * ((u * (4 * c_ipj - 3 * c_ij - c_ip2j) / (2 * dx)) +
                                       (v * (4 * c_ijp - 3 * c_ij - c_ijp2) / (2 * dx)));
            }

            sim->C_new[i][j] = c_ij + advection_term;
        }
    }

    double **temp = sim->C;
    sim->C = sim->C_new;
    sim->C_new = temp;
}


void first_order_upwind(Simulation *sim) {
    int Nx = sim->Nx;
    int Ny = sim->Ny;
    double dx = sim->dx;
    double dt = sim->dt;


    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(sim, Nx, Ny, dx, dt)
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            double x = -sim->L / 2.0 + (sim->global_start_x + i - 1) * dx;
            double y = -sim->L / 2.0 + (sim->global_start_y + j - 1) * dx;

            double u = sqrt(2.0) * y;  
            double v = -sqrt(2.0) * x; 

            double c_imj = sim->C[i - 1][j];
            double c_ipj = sim->C[i + 1][j];
            double c_ijm = sim->C[i][j - 1];
            double c_ijp = sim->C[i][j + 1];
            double c_ij = sim->C[i][j];

            double advection_term = 0.0;

            if (u > 0 && v > 0) {
                advection_term = dt * ((u * (c_ij - c_imj) / dx) +
                                       (v * (c_ij - c_ijm) / dx));
            } else if (u > 0 && v < 0) {
                advection_term = dt * ((u * (c_ij - c_imj) / dx) +
                                       (v * (c_ijp - c_ij) / dx));
            } else if (u < 0 && v > 0) {
                advection_term = dt * ((u * (c_ipj - c_ij) / dx) +
                                       (v * (c_ij - c_ijm) / dx));
            } else if (u < 0 && v < 0) {
                advection_term = dt * ((u * (c_ipj - c_ij) / dx) +
                                       (v * (c_ijp - c_ij) / dx));
            }

            sim->C_new[i][j] = c_ij - advection_term;
        }
    }

    double **temp = sim->C;
    sim->C = sim->C_new;
    sim->C_new = temp;
}

void lax_update(Simulation *sim) {
    int Nx = sim->Nx;
    int Ny = sim->Ny;
    double dx = sim->dx;
    double dt = sim->dt;

    double advection_coeff = -(dt / (2.0 * dx));

    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(sim, Nx, Ny, dx, dt, advection_coeff)
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {

            double x = -sim->L / 2.0 + (sim->global_start_x + i - 1) * dx;
            double y = -sim->L / 2.0 + (sim->global_start_y + j - 1) * dx;

            double u =  sqrt(2.0) * y;
            double v = -sqrt(2.0) * x;

            double c_imj = sim->C[i - 1][j];
            double c_ipj = sim->C[i + 1][j];
            double c_ijm = sim->C[i][j - 1];
            double c_ijp = sim->C[i][j + 1];

            double lax_avg_neighbors = (c_imj + c_ipj + c_ijm + c_ijp) / 4.0;

            double advection_term = advection_coeff * 
                                   (u * (c_ipj - c_imj) + v * (c_ijp - c_ijm));

            sim->C_new[i][j] = lax_avg_neighbors + advection_term;
        }
    }

    double **temp = sim->C;
    sim->C = sim->C_new;
    sim->C_new = temp;
}

void save_to_binary_with_global_coords(Simulation *sim, int Nx, int Ny, int global_start_x, int global_start_y, int step, const char *output_dir) {
    char filename[256];
    sprintf(filename, "%s/output_rank_%d_step_%d.bin", output_dir, sim->rank, step); 

    FILE *file = fopen(filename, "wb"); 
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s for writing.\n", filename);
        return;
    }

    int gl = sim->ghost_layers;
    for (int i = gl; i <= Nx + gl - 1; i++) {
        for (int j = gl; j <= Ny + gl - 1; j++) {
            double x = -sim->L / 2.0 + (global_start_x + i) * sim->dx;
            double y = -sim->L / 2.0 + (global_start_y + j) * sim->dx;
            double val = sim->C[i][j];
            fwrite(&x,   sizeof(double), 1, file);
            fwrite(&y,   sizeof(double), 1, file);
            fwrite(&val, sizeof(double), 1, file);
        }
    }

    fclose(file); 
}

void free_simulation(Simulation *sim) {
    if (sim->C != NULL) {
        free_dmatrix(sim->C, 0, sim->Nx_alloc, 0, sim->Ny_alloc);
        sim->C = NULL;
    }
    if (sim->C_new != NULL) {
        free_dmatrix(sim->C_new, 0, sim->Nx_alloc, 0, sim->Ny_alloc);
        sim->C_new = NULL;
    }
}
