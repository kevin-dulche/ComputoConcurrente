#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define MAX_CITIES 1002
#define NUM_ANTS 5000
#define MAX_ITER 200
#define ALPHA 1.0f
#define BETA 5.0f
#define RHO 0.5f
#define QVAL 100.0f

__device__ __host__ float euclidean(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

__global__ void init_rng(curandState *states, int seed) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < NUM_ANTS) {
        curand_init(seed, id, 0, &states[id]);
    }
}

__global__ void compute_distances(float *x, float *y, float *dist_matrix, float *pheromone, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        // Cada hilo calcula la distancia desde la ciudad idx a todas las demás
        for (int j = idx; j < n; j++) {
            float d = euclidean(x[idx], y[idx], x[j], y[j]);
            dist_matrix[idx * n + j] = d;
            dist_matrix[j * n + idx] = d;  // Aprovechamos la simetría de la matriz
            pheromone[idx * n + j] = 1.0f;
            pheromone[j * n + idx] = 1.0f;  // Matriz simétrica de feromonas
        }
    }
}


__device__ int select_next_city(int current, bool *visited, float *pheromone, float *dist_matrix, int n, curandState *state) {
    float sum = 0.0f;
    float prob[MAX_CITIES];

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            float tau = powf(pheromone[current * n + i], ALPHA);
            float eta = powf(1.0f / dist_matrix[current * n + i], BETA);
            prob[i] = tau * eta;
            sum += prob[i];
        } else {
            prob[i] = 0.0f;
        }
    }

    float r = curand_uniform(state) * sum;
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        total += prob[i];
        if (total >= r)
            return i;
    }

    for (int i = 0; i < n; i++)
        if (!visited[i])
            return i;

    return 0;
}

__global__ void construct_tours(int *tours, float *pheromone, float *dist_matrix, int n, curandState *states) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    if (k >= NUM_ANTS) return;

    bool visited[MAX_CITIES] = {0};
    curandState local_state = states[k];

    int start = curand(&local_state) % n;
    tours[k * n] = start;
    visited[start] = true;

    for (int i = 1; i < n; i++) {
        int next = select_next_city(tours[k * n + i - 1], visited, pheromone, dist_matrix, n, &local_state);
        tours[k * n + i] = next;
        visited[next] = true;
    }

    states[k] = local_state;
}

__global__ void compute_distances_of_tours(int *tours, float *dist_matrix, float *lengths, int n) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    if (k >= NUM_ANTS) return;

    float d = 0.0f;
    for (int i = 0; i < n - 1; i++) {
        int from = tours[k * n + i];
        int to = tours[k * n + i + 1];
        d += dist_matrix[from * n + to];
    }
    d += dist_matrix[tours[k * n + n - 1] * n + tours[k * n]];
    lengths[k] = d;
}

__global__ void evaporate_pheromones(float *pheromone, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;  // Total de celdas en la matriz de feromonas

    if (idx < total) {
        pheromone[idx] *= (1.0f - RHO);
    }
}


__global__ void deposit_pheromones(int *tours, float *lengths, float *pheromone, int n) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    if (k >= NUM_ANTS) return;

    float contrib = QVAL / lengths[k];
    for (int i = 0; i < n - 1; i++) {
        int from = tours[k * n + i];
        int to = tours[k * n + i + 1];
        atomicAdd(&pheromone[from * n + to], contrib);
        atomicAdd(&pheromone[to * n + from], contrib);
    }
    int last = tours[k * n + n - 1];
    int first = tours[k * n];
    atomicAdd(&pheromone[last * n + first], contrib);
    atomicAdd(&pheromone[first * n + last], contrib);
}

int main() {
    // FILE *fp = fopen("berlin52.tsp", "r");
    FILE *fp = fopen("pr1002.tsp", "r");
    if (!fp) {
        printf("No se pudo abrir el archivo.\n");
        return 1;
    }

    float *x_h = (float *)malloc(MAX_CITIES * sizeof(float));
    float *y_h = (float *)malloc(MAX_CITIES * sizeof(float));
    int num_cities = 0;
    char line[128];

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "NODE_COORD_SECTION", 18) == 0)
            break;
    }

    int index;
    float x, y;
    while (fscanf(fp, "%d %f %f", &index, &x, &y) == 3) {
        x_h[num_cities] = x;
        y_h[num_cities] = y;
        num_cities++;
    }
    fclose(fp);


    // Reservar memoria en GPU
    float *x_d, *y_d, *dist_matrix_d, *pheromone_d, *lengths_d;
    int *tours_d;
    curandState *states; /*Este puntero apunta a un arreglo de estados del generador aleatorio CUDA (cuRAND).
                            Se usa para generar números aleatorios en cada hilo de GPU.
                            Cada hormiga/hilo necesita su propio curandState.*/

    size_t sz = num_cities * num_cities * sizeof(float);
    cudaMalloc(&x_d, num_cities * sizeof(float));
    cudaMalloc(&y_d, num_cities * sizeof(float));
    cudaMalloc(&dist_matrix_d, sz);
    cudaMalloc(&pheromone_d, sz);
    cudaMalloc(&tours_d, NUM_ANTS * num_cities * sizeof(int));
    cudaMalloc(&lengths_d, NUM_ANTS * sizeof(float));
    cudaMalloc(&states, NUM_ANTS * sizeof(curandState));

    cudaMemcpy(x_d, x_h, num_cities * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, num_cities * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 0 es el ID del dispositivo (la primera GPU en el sistema)

    // Asegúrate de que el número de hilos por bloque no sea mayor que el número de hormigas
    // --- Antes del bucle o donde definas blocks/threadsPerBlock para las hormigas ---
    int threadsPerBlockAnts = min(prop.maxThreadsPerBlock, NUM_ANTS); // Renombrado para claridad
    int blocksAnts = (NUM_ANTS + threadsPerBlockAnts - 1) / threadsPerBlockAnts; // Renombrado

    // --- Configuración específica para la evaporación ---
    long long total_pheromone_elements = (long long)num_cities * num_cities;
    int threadsPerBlockEvap = prop.maxThreadsPerBlock; // O 512, etc.
    // Asegúrate de que threadsPerBlockEvap <= prop.maxThreadsPerBlock
    threadsPerBlockEvap = min(threadsPerBlockEvap, prop.maxThreadsPerBlock);
    int blocksEvap = (total_pheromone_elements + threadsPerBlockEvap - 1) / threadsPerBlockEvap;

    printf("Calculando distancias...\n");
    compute_distances<<<blocksAnts, threadsPerBlockAnts>>>(x_d, y_d, dist_matrix_d, pheromone_d, num_cities);

    // init_rng<<<(NUM_ANTS + 255) / 256, 256>>>(states, time(NULL));
    // Llamada al kernel con los bloques y hilos calculados
    printf("Inicializando generadores aleatorios...\n");
    init_rng<<<blocksAnts, threadsPerBlockAnts>>>(states, time(NULL));

    float best_length = 1e9;
    int *best_tour_h = (int *)malloc(num_cities * sizeof(int));
    int *tours_h = (int *)malloc(NUM_ANTS * num_cities * sizeof(int));
    float *lengths_h = (float *)malloc(NUM_ANTS * sizeof(float));

    printf("Iniciando el algoritmo...\n");
    clock_t inicio, final;
    double tiempo_total;
    inicio = clock();
    for (int iter = 0; iter < MAX_ITER; iter++) {
        construct_tours<<<blocksAnts, threadsPerBlockAnts>>>(tours_d, pheromone_d, dist_matrix_d, num_cities, states);
        compute_distances_of_tours<<<blocksAnts, threadsPerBlockAnts>>>(tours_d, dist_matrix_d, lengths_d, num_cities);

        evaporate_pheromones<<<blocksEvap, threadsPerBlockEvap>>>(pheromone_d, num_cities);
        deposit_pheromones<<<blocksAnts, threadsPerBlockAnts>>>(tours_d, lengths_d, pheromone_d, num_cities);

        cudaMemcpy(lengths_h, lengths_d, NUM_ANTS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tours_h, tours_d, NUM_ANTS * num_cities * sizeof(int), cudaMemcpyDeviceToHost);

        for (int k = 0; k < NUM_ANTS; k++) {
            if (lengths_h[k] < best_length) {
                best_length = lengths_h[k];
                memcpy(best_tour_h, &tours_h[k * num_cities], num_cities * sizeof(int));
            }
        }
        printf("Iteración %d, Mejor longitud actual: %.2f\n", iter + 1, best_length);
    }
    final = clock();
    tiempo_total = (double)(final - inicio) / CLOCKS_PER_SEC;
    printf("Tiempo total: %.2f segundos\n", tiempo_total);

    printf("Mejor tour encontrado (CUDA Ant System): %.2f\n", best_length);
    for (int i = 0; i < num_cities; i++)
        printf("%d->", best_tour_h[i] + 1);
    printf("%d\n", best_tour_h[0] + 1);

    // Liberar memoria
    cudaFree(x_d); cudaFree(y_d); cudaFree(dist_matrix_d);
    cudaFree(pheromone_d); cudaFree(tours_d); cudaFree(lengths_d); cudaFree(states);
    free(x_h); free(y_h); free(best_tour_h); free(tours_h); free(lengths_h);

    return 0;
}
