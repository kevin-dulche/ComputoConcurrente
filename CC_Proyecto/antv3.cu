
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_CITIES 1002
#define NUM_ANTS 50
#define MAX_ITER 200
#define ALPHA 1.0
#define BETA 5.0
#define RHO 0.5
#define QVAL 100.0

typedef struct {
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
    double *pheromone;
    double *dist_matrix;
} TSPData;



__global__ void compute_distances_kernel(TSPData *data, double *d_dist_matrix, double *d_pheromone) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < data->num_cities && j < data->num_cities) {
        double dx = data->x[i] - data->x[j];
        double dy = data->y[i] - data->y[j];
        d_dist_matrix[i * data->num_cities + j] = sqrt(dx * dx + dy * dy);
        d_pheromone[i * data->num_cities + j] = 1.0;
    }
}

__global__ void ant_kernel(TSPData *data, int *d_ants, int *d_visited, int *d_best_tour, double *d_best_len) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < NUM_ANTS) {
        // Memoria local para cada hormiga (visited)
        int visited[MAX_CITIES] = {0};
        
        // Inicia la primera ciudad de la hormiga
        d_ants[k * data->num_cities] = rand() % data->num_cities;
        visited[d_ants[k * data->num_cities]] = 1;
        
        // Recorrer las ciudades
        for (int i = 1; i < data->num_cities; i++) {
            d_ants[k * data->num_cities + i] = select_next(d_ants[k * data->num_cities + i - 1], visited, data->num_cities, data->pheromone, data->dist_matrix);
            visited[d_ants[k * data->num_cities + i]] = 1;
        }
        
        // Calcular la distancia total de la ruta
        double d = total_distance(&d_ants[k * data->num_cities], data->num_cities);
        
        // Actualizar la mejor longitud y el mejor tour
        if (d < *d_best_len) {
            *d_best_len = d;
            for (int i = 0; i < data->num_cities; i++) {
                d_best_tour[i] = d_ants[k * data->num_cities + i];
            }
        }
    }
}

__device__ int select_next(int current_city, int *visited, int num_cities, double *pheromone, double *dist_matrix) {
    double total_prob = 0.0;
    double probs[MAX_CITIES];
    
    // Calcular las probabilidades de cada ciudad no visitada
    for (int i = 0; i < num_cities; i++) {
        if (visited[i] == 0) {  // Si la ciudad no ha sido visitada
            double pheromone_val = pheromone[current_city * num_cities + i];  // Lectura de memoria global
            double distance_val = dist_matrix[current_city * num_cities + i];  // Lectura de memoria global
            probs[i] = pheromone_val / (distance_val + 1e-6);  // Evitar división por cero
            total_prob += probs[i];
        } else {
            probs[i] = 0.0;
        }
    }

    // Selección de la siguiente ciudad basado en las probabilidades
    double rand_val = (double)rand() / RAND_MAX * total_prob;
    double cumulative_prob = 0.0;
    
    for (int i = 0; i < num_cities; i++) {
        if (visited[i] == 0) {
            cumulative_prob += probs[i];
            if (cumulative_prob >= rand_val) {
                return i;  // Selecciona la ciudad
            }
        }
    }
    
    return -1;  // En caso de error
}


void compute_distances(TSPData *data) {
    // Allocate memory on device
    double *d_dist_matrix, *d_pheromone;
    TSPData *d_data;

    size_t matrix_size = data->num_cities * data->num_cities * sizeof(double);

    // Allocate memory on device for the TSPData and matrices
    cudaMalloc((void **)&d_data, sizeof(TSPData));
    cudaMalloc((void **)&d_dist_matrix, matrix_size);
    cudaMalloc((void **)&d_pheromone, matrix_size);

    // Copy data to device
    cudaMemcpy(d_data, data, sizeof(TSPData), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(16, 16); // Block size 16x16 (adjustable)
    dim3 gridSize((data->num_cities + blockSize.x - 1) / blockSize.x,
                    (data->num_cities + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    compute_distances_kernel<<<gridSize, blockSize>>>(d_data, d_dist_matrix, d_pheromone);

    // Check for any CUDA errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Copy results back to host
    cudaMemcpy(data->dist_matrix, d_dist_matrix, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(data->pheromone, d_pheromone, matrix_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_dist_matrix);
    cudaFree(d_pheromone);
}



double total_distance(int *tour, int n, double *dist_matrix) {
    double d = 0.0;
    for(int i = 0; i < n - 1; i++){
        d += dist_matrix[tour[i]][tour[i+1]];
    }
    d += dist_matrix[tour[n-1]][tour[0]];
    return d;
}




void ant_system(TSPData *data) {
    // Asignar memoria en el dispositivo
    int *d_ants, *d_visited, *d_best_tour;
    double *d_best_len;

    size_t ants_size = NUM_ANTS * data->num_cities * sizeof(int);
    cudaMalloc((void **)&d_ants, ants_size);
    cudaMalloc((void **)&d_best_tour, data->num_cities * sizeof(int));
    cudaMalloc((void **)&d_best_len, sizeof(double));

    // Inicializar el valor de la mejor longitud
    double best_len = 1e9;
    cudaMemcpy(d_best_len, &best_len, sizeof(double), cudaMemcpyHostToDevice);

    // Copiar los datos necesarios a la memoria del dispositivo
    TSPData *d_data;
    cudaMalloc((void **)&d_data, sizeof(TSPData));
    cudaMemcpy(d_data, data, sizeof(TSPData), cudaMemcpyHostToDevice);

    // Configuración de bloques y hilos
    dim3 blockSize(16);
    dim3 gridSize((NUM_ANTS + blockSize.x - 1) / blockSize.x);

    // Ejecutar el kernel
    for (int iter = 0; iter < MAX_ITER; iter++) {
        ant_kernel<<<gridSize, blockSize>>>(d_data, d_ants, d_visited, d_best_tour, d_best_len);

        // Sincronizar y verificar errores
        cudaDeviceSynchronize();

        // Copiar el resultado de vuelta al host
        cudaMemcpy(&best_len, d_best_len, sizeof(double), cudaMemcpyDeviceToHost);
        printf("Iteración %d, Mejor longitud actual: %.2f\n", iter + 1, best_len);
    }

    // Liberar memoria en el dispositivo
    cudaFree(d_ants);
    cudaFree(d_best_tour);
    cudaFree(d_best_len);
    cudaFree(d_data);
}


int main(){
    srand(time(NULL));
    TSPData data;
    FILE *fp = fopen("pr1002.tsp", "r");
    if(!fp){
        printf("No se pudo abrir el archivo.\n");
        return 1;
    }

    char line[128];
    while(fgets(line, sizeof(line), fp)){
        if(strncmp(line, "NODE_COORD_SECTION", 18) == 0)
            break;
    }

    int index;
    double x, y;
    data.num_cities = 0;
    while(fscanf(fp, "%d %lf %lf", &index, &x, &y) == 3){
        data.x[data.num_cities] = x;
        data.y[data.num_cities] = y;
        data.num_cities++;
    }
    fclose(fp);
    data.pheromone = (double *)malloc(data.num_cities * data.num_cities * sizeof(double));
    data.dist_matrix = (double *)malloc(data.num_cities * data.num_cities * sizeof(double));

    ant_system(&data);
    return 0;
}