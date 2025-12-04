#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX_CITIES 1002
#define NUM_ANTS 1024  // Múltiplo de 32 para mejor rendimiento en GPU
#define MAX_ITER 200
#define ALPHA 1.0
#define BETA 5.0
#define RHO 0.5
#define QVAL 100.0

typedef struct {
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
} TSPData;

// Kernel para Inicialización de Generadores Aleatorios
__global__ void init_rng(curandState *states, int seed) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < NUM_ANTS) {
        curand_init(seed, id, 0, &states[id]);
    }
}

// Kernel para Cálculo de Distancias
__global__ void compute_distances_kernel(double *d_x, double *d_y, double *d_dist_matrix, int num_cities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < num_cities && j < num_cities) {
        double dx = d_x[i] - d_x[j];
        double dy = d_y[i] - d_y[j];
        d_dist_matrix[i * num_cities + j] = sqrt(dx*dx + dy*dy);
    }
}

__global__ void compute_lengths_kernel(int *d_tours, double *d_dist_matrix, double *d_lengths, int num_cities) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(ant_id >= NUM_ANTS) return;
    
    double length = 0.0;
    for(int i = 0; i < num_cities - 1; i++) {
        int from = d_tours[ant_id * num_cities + i];
        int to = d_tours[ant_id * num_cities + i + 1];
        length += d_dist_matrix[from * num_cities + to];
    }
    // Añadir la conexión final-inicial
    length += d_dist_matrix[d_tours[ant_id * num_cities + num_cities - 1] * num_cities + d_tours[ant_id * num_cities]];
    d_lengths[ant_id] = length;
}


// Kernel para Construcción de Tours
__device__ int select_next_city(int current, int *visited, double *pheromone, double *dist_matrix, int num_cities, curandState *state) {
    double sum = 0.0;
    double probs[MAX_CITIES];
    int candidates[MAX_CITIES];
    int count = 0;
    
    for(int i = 0; i < num_cities; i++) {
        if(!visited[i]) {
            candidates[count] = i;
            probs[count] = pow(pheromone[current * num_cities + i], ALPHA) * 
                          pow(1.0 / dist_matrix[current * num_cities + i], BETA);
            sum += probs[count];
            count++;
        }
    }
    
    double r = curand_uniform_double(state) * sum;
    double accum = 0.0;
    for(int i = 0; i < count; i++) {
        accum += probs[i];
        if(accum >= r) return candidates[i];
    }
    return candidates[0];
}

__global__ void construct_tours_kernel(int *d_tours, double *d_pheromone, double *d_dist_matrix, 
                                     int num_cities, curandState *states) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(ant_id >= NUM_ANTS) return;
    
    curandState local_state = states[ant_id];
    int visited[MAX_CITIES] = {0};
    
    int start_city = curand(&local_state) % num_cities;
    d_tours[ant_id * num_cities] = start_city;
    visited[start_city] = 1;
    
    for(int i = 1; i < num_cities; i++) {
        int next = select_next_city(d_tours[ant_id * num_cities + i - 1], visited, d_pheromone, d_dist_matrix, num_cities, &local_state);
        d_tours[ant_id * num_cities + i] = next;
        visited[next] = 1;
    }
    
    states[ant_id] = local_state;
}

// Kernels para Actualización de Feromonas
__global__ void evaporate_pheromones_kernel(double *d_pheromone, int num_cities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_cities * num_cities) {
        d_pheromone[idx] *= (1.0 - RHO);
    }
}

__global__ void deposit_pheromones_no_atomic(int *d_tours, double *d_lengths, double *d_pheromone, int num_cities) {
    extern __shared__ double local_pheromone[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Inicializar memoria compartida
    for (int i = tid; i < num_cities * num_cities; i += blockDim.x) {
        local_pheromone[i] = 0.0;
    }
    __syncthreads();
    
    // Cada hormiga en el bloque acumula en local
    for (int ant = bid; ant < NUM_ANTS; ant += gridDim.x) {
        double contrib = QVAL / d_lengths[ant];
        for (int i = 0; i < num_cities - 1; i++) {
            int from = d_tours[ant * num_cities + i];
            int to = d_tours[ant * num_cities + i + 1];
            local_pheromone[from * num_cities + to] += contrib;
            local_pheromone[to * num_cities + from] += contrib;
        }
    }
    __syncthreads();
    
    // Fusionar resultados globales (sin atómicas)
    for (int i = tid; i < num_cities * num_cities; i += blockDim.x) {
        if (local_pheromone[i] > 0) {
            d_pheromone[i] += local_pheromone[i];
        }
    }
}

void ant_system_gpu(TSPData *data) {
    // 1. Reservar memoria en GPU
    double *d_x, *d_y, *d_dist_matrix, *d_pheromone, *d_lengths;
    int *d_tours;
    curandState *d_states;
    
    printf("Reservando memoria en GPU...\n");
    cudaMalloc(&d_x, data->num_cities * sizeof(double));
    cudaMalloc(&d_y, data->num_cities * sizeof(double));
    cudaMalloc(&d_dist_matrix, data->num_cities * data->num_cities * sizeof(double));
    cudaMalloc(&d_pheromone, data->num_cities * data->num_cities * sizeof(double));
    cudaMalloc(&d_tours, NUM_ANTS * data->num_cities * sizeof(int));
    cudaMalloc(&d_lengths, NUM_ANTS * sizeof(double));
    cudaMalloc(&d_states, NUM_ANTS * sizeof(curandState));
    
    // 2. Copiar datos a GPU
    printf("Copiando datos a GPU...\n");
    cudaMemcpy(d_x, data->x, data->num_cities * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, data->y, data->num_cities * sizeof(double), cudaMemcpyHostToDevice);
    
    // 3. Inicializar feromonas y estados aleatorios
    printf("Inicializando feromonas y estados aleatorios...\n");
    dim3 blockSize(16, 16);
    dim3 gridSize((data->num_cities + blockSize.x - 1) / blockSize.x, (data->num_cities + blockSize.y - 1) / blockSize.y);
    
    // Inicializar matriz de feromonas a 1.0
    printf("Inicializando matriz de feromonas...\n");
    cudaMemset(d_pheromone, 1.0, data->num_cities * data->num_cities * sizeof(double));
    
    // Inicializar generadores aleatorios
    printf("Inicializando generadores aleatorios...\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_ANTS + threadsPerBlock - 1) / threadsPerBlock;
    init_rng<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL));
    
    // 4. Calcular matriz de distancias
    printf("Calculando matriz de distancias...\n");
    compute_distances_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_dist_matrix, data->num_cities);
    
    // 5. Bucle principal
    int best_tour[MAX_CITIES];
    double best_len = 1e9;
    double *h_lengths = (double*)malloc(NUM_ANTS * sizeof(double));
    int *h_tours = (int*)malloc(NUM_ANTS * data->num_cities * sizeof(int));
    
    printf("Iniciando el algoritmo...\n");
    for(int iter = 0; iter < MAX_ITER; iter++) {
        // Construir tours
        printf("Construyendo tours...\n");
        construct_tours_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tours, d_pheromone, d_dist_matrix, data->num_cities, d_states);
        
        // Calcular longitudes
        printf("Calculando longitudes...\n");
        compute_lengths_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tours, d_dist_matrix, d_lengths, data->num_cities);
        
        // Actualizar feromonas
        printf("Actualizando feromonas...\n");
        evaporate_pheromones_kernel<<<(data->num_cities*data->num_cities+255)/256, 256>>>(d_pheromone, data->num_cities);
        size_t shared_mem_size = MAX_CITIES * MAX_CITIES * sizeof(double);
        deposit_pheromones_no_atomic<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_tours, d_lengths, d_pheromone, data->num_cities);
        
        if (iter == MAX_ITER - 1) {
            printf("Esperando a que todos los hilos terminen...\n");
            cudaDeviceSynchronize();
        }
        // Copiar resultados a CPU para encontrar el mejor
        printf("Copiando resultados a CPU...\n");
        cudaMemcpy(h_lengths, d_lengths, NUM_ANTS * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tours, d_tours, NUM_ANTS * data->num_cities * sizeof(int), cudaMemcpyDeviceToHost);
        
        for(int k = 0; k < NUM_ANTS; k++) {
            if(h_lengths[k] < best_len) {
                best_len = h_lengths[k];
                memcpy(best_tour, &h_tours[k * data->num_cities], data->num_cities * sizeof(int));
            }
        }
        
        // if(iter % 10 == 0) {
        //     printf("Iteración %d, Mejor longitud: %.2f\n", iter, best_len);
        // }
        printf("Iteración %d, Mejor longitud: %.2f\n", iter, best_len);
    }
    
    printf("Mejor tour encontrado (GPU): %.2f\n", best_len);
    for(int i = 0; i < data->num_cities; i++) {
        printf("%d->", best_tour[i] + 1);
    }
    printf("%d\n", best_tour[0] + 1);
    
    // Liberar memoria
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_dist_matrix);
    cudaFree(d_pheromone); cudaFree(d_tours); cudaFree(d_lengths); cudaFree(d_states);
    free(h_lengths); free(h_tours);
}

int main(){
    srand(time(NULL));
    TSPData data;
    FILE *fp = fopen("pr1002.tsp", "r");
    if(!fp){
        printf("No se pudo abrir el archivo.\n");
        return 1;
    }

    printf("Leyendo archivo...\n");
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

    // Llamar a la función del sistema de hormigas
    ant_system_gpu(&data);

    return 0;
}