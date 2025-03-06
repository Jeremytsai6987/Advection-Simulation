#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>

typedef struct {
    int N;            
    double L;         
    double dx;        
    double dt;        
    int NT; 
    int Nx;
    int Ny;
    int Nx_alloc;
    int Ny_alloc;
    int rank;
    int global_start_x; // Global start x-coordinate of this subdomain
    int global_start_y; // Global start y-coordinate of this subdomain        
    double **C;       
    double **C_new;   
    int ghost_layers;  

} Simulation;


void initialize_simulation(Simulation *sim, int Nx, int Ny, int global_start_x, int global_start_y, double L, double T, int method);
void update_double_ghost_cells(Simulation *sim);
void update_ghost_cells(Simulation *sim);
void exchange_ghost_cells_mpi(Simulation *sim, int global_start_x, int global_start_y, int neighbors[4]);
void lax_update(Simulation *sim);
void save_to_binary_with_global_coords(Simulation *sim, int Nx, int Ny, int global_start_x, int global_start_y, int step, const char *output_dir);
void free_simulation(Simulation *sim);
void first_order_upwind(Simulation *sim);
void second_order_upwind(Simulation *sim);
#endif


