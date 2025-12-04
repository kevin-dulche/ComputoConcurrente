#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits> // Para std::numeric_limits

#define MAX_CITIES 1002 // Asegúrate que sea suficiente para pr1002.tsp
#define NUM_ANTS 1000
#define MAX_ITER 200
#define ALPHA 1.0
#define BETA 5.0
#define RHO 0.5    // Tasa de evaporación
#define QVAL 100.0 // Cantidad de feromona

// --- Comprobación de Errores CUDA ---
#define CHECK_CUDA_ERROR(err)                                                                      \
    if (err != cudaSuccess)                                                                        \
    {                                                                                              \
        fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                        \
    }

#define CHECK_CURAND_ERROR(err)                                         \
    if (err != CURAND_STATUS_SUCCESS)                                   \
    {                                                                   \
        fprintf(stderr, "Error cuRAND en %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                             \
    }
// --- Fin Comprobación de Errores CUDA ---

// --- Estructuras de Datos ---
typedef struct
{
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
} TSPData;

// --- Kernels CUDA ---

// Kernel para calcular la matriz de distancias e inicializar feromonas
__global__ void compute_distances_kernel(double *d_dist_matrix, float *d_pheromone, // <-- CAMBIO A FLOAT
                                         const double *d_x, const double *d_y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        if (i == j)
        {
            d_dist_matrix[i * n + j] = 0.0; // Distancia a sí misma es 0
            // d_pheromone[i * n + j] = 0.0;   // No hay feromona en el mismo nodo
            d_pheromone[i * n + j] = 0.0f; // <-- CAMBIO A FLOAT
        }
        else
        {
            double dx = d_x[i] - d_x[j];
            double dy = d_y[i] - d_y[j];
            double dist = sqrt(dx * dx + dy * dy);
            d_dist_matrix[i * n + j] = dist;
            // Inicializar feromona (valor pequeño > 0 para evitar división por cero en select_next si dist=0 es posible teóricamente)
            // d_pheromone[i * n + j] = 1.0 / (double)n; // O alguna otra heurística inicial
            d_pheromone[i * n + j] = 1.0f / (float)n; // <-- CAMBIO A FLOAT
        }
    }
}

// Kernel para la evaporación de feromonas
__global__ void evaporate_pheromone_kernel(float *d_pheromone, int n, double rho)  // <-- CAMBIO A FLOAT
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n && i != j)
    {
        d_pheromone[i * n + j] *= (float)(1.0 - rho); // <-- CAMBIO A FLOAT
    }
}

// Kernel para que cada hormiga construya una ruta
// ¡Este es el kernel más complejo!
__global__ void construct_tours_kernel(const double *d_dist_matrix, const float *d_pheromone, // <-- CAMBIO A FLOAT
                                       int *d_ants_tours, curandState *rand_states, int n,
                                       double alpha, double beta, bool *d_visited_flags) // Pasar memoria para visited
{
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (ant_id >= NUM_ANTS)
        return; // Verificar si el hilo es válido para una hormiga

    curandState local_rand_state = rand_states[ant_id]; // Cargar estado del generador

    // Usar la memoria d_visited_flags: cada hormiga tiene su propia sección
    bool *my_visited = d_visited_flags + ant_id * n;

    // 1. Inicializar 'visited' para esta hormiga
    for (int i = 0; i < n; ++i)
    {
        my_visited[i] = false;
    }

    // 2. Elegir ciudad inicial aleatoria para la hormiga 'ant_id'
    int start_node = curand(&local_rand_state) % n;
    d_ants_tours[ant_id * n + 0] = start_node;
    my_visited[start_node] = true;
    int current_city = start_node;

    // 3. Construir el resto de la ruta (n-1 pasos)
    for (int step = 1; step < n; ++step)
    {
        double total_prob = 0.0;
        double probabilities[MAX_CITIES]; // Podría ser demasiado grande para registros/shared mem si n es grande

        // Calcular probabilidades de transición a ciudades no visitadas
        for (int next_city = 0; next_city < n; ++next_city)
        {
            if (!my_visited[next_city])
            {
                // Asegurarse de que dist_matrix[current][next] no sea cero
                double dist = d_dist_matrix[current_city * n + next_city];
                float pher = d_pheromone[current_city * n + next_city]; // <-- Leer como FLOAT

                // Evitar división por cero o valores inválidos
                if (dist < 1e-9) dist = 1e-9; // Distancia mínima pequeña
                if (pher < 1e-9) pher = 1e-9f; // Feromona mínima pequeña

                double prob = pow(pher, alpha) * pow(1.0 / dist, beta);
                probabilities[next_city] = prob;
                total_prob += prob;
            }
            else
            {
                probabilities[next_city] = 0.0;
            }
        }

        // Seleccionar la siguiente ciudad usando la ruleta
        int selected_city = -1;
        if (total_prob <= 1e-9)
        {
            // Caso raro: no hay opciones válidas (quizás todas visitadas o probs cero)
            // Escoger la primera no visitada que encuentre
            for (int i = 0; i < n; ++i)
            {
                if (!my_visited[i])
                {
                    selected_city = i;
                    break;
                }
            }
            // Si aún así no encuentra (no debería pasar en TSP), forzar una
            if (selected_city == -1)
                selected_city = (current_city + 1) % n;
        }
        else
        {
            double rand_val = curand_uniform_double(&local_rand_state) * total_prob;
            double cumulative_prob = 0.0;
            for (int next_city = 0; next_city < n; ++next_city)
            {
                if (!my_visited[next_city] && probabilities[next_city] > 0)
                {
                    cumulative_prob += probabilities[next_city];
                    if (cumulative_prob >= rand_val)
                    {
                        selected_city = next_city;
                        break;
                    }
                }
            }
            // Si por algún error de precisión no se eligió, elegir la última opción válida
            if (selected_city == -1)
            {
                for (int i = n - 1; i >= 0; --i)
                {
                    if (!my_visited[i] && probabilities[i] > 0)
                    {
                        selected_city = i;
                        break;
                    }
                }
            }
            // Fallback extremo
            if (selected_city == -1)
            {
                for (int i = 0; i < n; ++i)
                {
                    if (!my_visited[i])
                    {
                        selected_city = i;
                        break;
                    }
                }
                if (selected_city == -1)
                    selected_city = (current_city + 1) % n;
            }
        }

        // Mover a la ciudad seleccionada
        d_ants_tours[ant_id * n + step] = selected_city;
        my_visited[selected_city] = true;
        current_city = selected_city;
    }

    // Guardar el estado del generador para la próxima iteración
    rand_states[ant_id] = local_rand_state;
}

// Kernel para calcular la longitud de la ruta de cada hormiga
__global__ void calculate_distances_kernel(const double *d_dist_matrix, const int *d_ants_tours,
                                           double *d_ant_distances, int n)
{
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (ant_id >= NUM_ANTS)
        return;

    double current_dist = 0.0;
    const int *tour = d_ants_tours + ant_id * n; // Puntero al inicio de la ruta de esta hormiga

    for (int i = 0; i < n - 1; ++i)
    {
        int from = tour[i];
        int to = tour[i + 1];
        current_dist += d_dist_matrix[from * n + to];
    }
    // Añadir la distancia de vuelta al inicio
    current_dist += d_dist_matrix[tour[n - 1] * n + tour[0]];

    d_ant_distances[ant_id] = current_dist;
}

// Kernel para depositar feromonas usando operaciones atómicas (VERSIÓN FLOAT)
__global__ void deposit_pheromone_kernel(float *d_pheromone, // <--- CAMBIO A FLOAT
                                         const int *d_ants_tours,
                                         const double *d_ant_distances, // Distancias pueden seguir siendo double
                                         int n, float qval)             // <--- CAMBIO A FLOAT
{
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (ant_id >= NUM_ANTS)
        return;

    const int *tour = d_ants_tours + ant_id * n;
    double tour_len = d_ant_distances[ant_id]; // Lee la distancia (double)

    // Evitar división por cero si la longitud es 0 (no debería pasar)
    if (tour_len < 1e-9)
        return;

    // Calcula el depósito como float
    float pheromone_deposit = (float)(qval / tour_len); // <--- CAMBIO A FLOAT y CAST

    for (int i = 0; i < n - 1; ++i)
    {
        int from = tour[i];
        int to = tour[i + 1];
        // Usar atomicAdd para actualizar feromonas de forma segura (ahora con float)
        atomicAdd(&d_pheromone[from * n + to], pheromone_deposit); // Funciona con float*
        atomicAdd(&d_pheromone[to * n + from], pheromone_deposit); // Funciona con float*
    }
    // Depósito en el arco de vuelta al inicio
    int last = tour[n - 1];
    int first = tour[0];
    atomicAdd(&d_pheromone[last * n + first], pheromone_deposit); // Funciona con float*
    atomicAdd(&d_pheromone[first * n + last], pheromone_deposit); // Funciona con float*
}

__global__ void init_curand_kernel(curandState *states, unsigned long long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_ANTS)
    {
        // Usar id y seed para inicializar cada estado de forma única
        curand_init(seed, id, 0, &states[id]);
    }
}

// --- Función Principal Host ---

void ant_system_cuda(TSPData *data)
{
    int n = data->num_cities;
    printf("Número de ciudades: %d\n", n);
    printf("Número de hormigas: %d\n", NUM_ANTS);
    printf("Número de iteraciones: %d\n", MAX_ITER);

    // --- Memoria del Host ---
    int *h_best_tour = (int *)malloc(n * sizeof(int));
    double h_best_len = std::numeric_limits<double>::max();
    double *h_ant_distances = (double *)malloc(NUM_ANTS * sizeof(double));
    if (!h_best_tour || !h_ant_distances)
    {
        fprintf(stderr, "Error al alocar memoria del host.\n");
        exit(EXIT_FAILURE);
    }

    // --- Memoria del Dispositivo ---
    double *d_x, *d_y, *d_dist_matrix; // Estos pueden seguir siendo double
    float *d_pheromone;               // <--- CAMBIO A FLOAT
    int *d_ants_tours, *d_best_tour;
    double *d_ant_distances;          // Puede seguir siendo double
    curandState *d_rand_states;
    bool *d_visited_flags;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dist_matrix, (size_t)n * n * sizeof(double)));
    // CHECK_CUDA_ERROR(cudaMalloc(&d_pheromone, (size_t)n * n * sizeof(double)));
    // Ajustar tamaño para float
    CHECK_CUDA_ERROR(cudaMalloc(&d_pheromone, (size_t)n * n * sizeof(float))); // <--- CAMBIO A FLOAT Y SIZEOF
    CHECK_CUDA_ERROR(cudaMalloc(&d_ants_tours, (size_t)NUM_ANTS * n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_best_tour, n * sizeof(int))); // Para guardar la mejor ruta en GPU si se desea
    CHECK_CUDA_ERROR(cudaMalloc(&d_ant_distances, NUM_ANTS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_rand_states, NUM_ANTS * sizeof(curandState)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_visited_flags, (size_t)NUM_ANTS * n * sizeof(bool))); // Alocar memoria para visited

    // --- Transferir datos iniciales al Dispositivo ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, data->x, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, data->y, n * sizeof(double), cudaMemcpyHostToDevice));

    // --- Configuración de Grid/Blocks ---
    // Para matrices NxN
    dim3 threadsPerBlock2D(16, 16);
    dim3 numBlocks2D((n + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                        (n + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);

    // Para operaciones por hormiga
    dim3 threadsPerBlock1D(256); // Ajustable
    dim3 numBlocks1D((NUM_ANTS + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);

    // --- Inicializar cuRAND ---
    // Kernel para inicializar los estados de cuRAND

    init_curand_kernel<<<numBlocks1D, threadsPerBlock1D>>>(d_rand_states, time(NULL));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Asegurarse que la inicialización termine

    // --- Calcular Distancias e Inicializar Feromonas (una vez) ---
    compute_distances_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_dist_matrix, d_pheromone, d_x, d_y, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Esperar a que termine

    // --- Bucle Principal de Iteraciones ACO ---
    printf("Iniciando iteraciones ACO en GPU...\n");
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {

        // 1. Construir rutas (cada hormiga en un hilo)
        construct_tours_kernel<<<numBlocks1D, threadsPerBlock1D>>>(d_dist_matrix, d_pheromone, d_ants_tours,
                                                                    d_rand_states, n, ALPHA, BETA, d_visited_flags);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Esperar a que todas las hormigas terminen

        // 2. Calcular la longitud de cada ruta construida
        calculate_distances_kernel<<<numBlocks1D, threadsPerBlock1D>>>(d_dist_matrix, d_ants_tours, d_ant_distances, n);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 3. Copiar longitudes de rutas del Device al Host para encontrar la mejor
        CHECK_CUDA_ERROR(cudaMemcpy(h_ant_distances, d_ant_distances, NUM_ANTS * sizeof(double), cudaMemcpyDeviceToHost));

        // Encontrar la mejor ruta de *esta* iteración en el Host
        double current_iter_best_len = h_ant_distances[0];
        int current_iter_best_ant_idx = 0;
        for (int k = 1; k < NUM_ANTS; ++k)
        {
            if (h_ant_distances[k] < current_iter_best_len)
            {
                current_iter_best_len = h_ant_distances[k];
                current_iter_best_ant_idx = k;
            }
        }

        // Actualizar la mejor ruta global encontrada hasta ahora
        if (current_iter_best_len < h_best_len)
        {
            h_best_len = current_iter_best_len;
            // Copiar la mejor ruta de esta iteración desde el Device al Host
            CHECK_CUDA_ERROR(cudaMemcpy(h_best_tour,
                                        d_ants_tours + (size_t)current_iter_best_ant_idx * n, // Offset correcto
                                        n * sizeof(int),
                                        cudaMemcpyDeviceToHost));
            printf("Iter %d: Nueva mejor longitud = %.2f\n", iter + 1, h_best_len);
        }
        else if ((iter + 1) % 10 == 0)
        { // Imprimir progreso ocasionalmente
            printf("Iter %d: Longitud actual = %.2f (Mejor global: %.2f)\n", iter + 1, current_iter_best_len, h_best_len);
        }

        // 4. Evaporar Feromonas
        evaporate_pheromone_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_pheromone, n, RHO);
        CHECK_CUDA_ERROR(cudaGetLastError());
        // No es estrictamente necesario sincronizar aquí si el depósito es la siguiente operación

        // 5. Depositar Feromonas (usando las rutas y longitudes calculadas)
        deposit_pheromone_kernel<<<numBlocks1D, threadsPerBlock1D>>>(d_pheromone, d_ants_tours, d_ant_distances, n, (float)QVAL);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Sincronizar después del depósito antes de la siguiente iteración

    } // Fin del bucle de iteraciones

    printf("\nMejor tour encontrado (CUDA ACO): %.2f\n", h_best_len);
    printf("Tour final:\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d->", h_best_tour[i] + 1); // +1 si los índices en el archivo .tsp empiezan desde 1
    }
    printf("%d\n", h_best_tour[0] + 1); // vuelta al inicio

    // --- Liberar Memoria ---
    free(h_best_tour);
    free(h_ant_distances);

    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_dist_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_pheromone));
    CHECK_CUDA_ERROR(cudaFree(d_ants_tours));
    CHECK_CUDA_ERROR(cudaFree(d_best_tour));
    CHECK_CUDA_ERROR(cudaFree(d_ant_distances));
    CHECK_CUDA_ERROR(cudaFree(d_rand_states));
    CHECK_CUDA_ERROR(cudaFree(d_visited_flags));
}

int main()
{
    // srand(time(NULL)); // No es necesario si cuRAND se inicializa con time(NULL)

    TSPData data;
    FILE *fp = fopen("pr1002.tsp", "r"); // Asegúrate que este archivo exista
    if (!fp)
    {
        printf("No se pudo abrir el archivo 'pr1002.tsp'.\n");
        return 1;
    }

    char line[256]; // Aumentado tamaño por si acaso
    int node_coord_section_found = 0;
    while (fgets(line, sizeof(line), fp))
    {
        if (strncmp(line, "DIMENSION", 9) == 0)
        {
            // Opcional: leer la dimensión si es necesario
            // sscanf(line, "DIMENSION : %d", &some_variable);
        }
        else if (strncmp(line, "NODE_COORD_SECTION", 18) == 0)
        {
            node_coord_section_found = 1;
            break;
        }
    }

    if (!node_coord_section_found)
    {
        printf("No se encontró 'NODE_COORD_SECTION' en el archivo.\n");
        fclose(fp);
        return 1;
    }

    int index;
    double x, y;
    data.num_cities = 0;
    // Leer las coordenadas
    while (fscanf(fp, "%d %lf %lf", &index, &x, &y) == 3)
    {
        if (data.num_cities >= MAX_CITIES)
        {
            printf("Error: Demasiadas ciudades en el archivo, aumenta MAX_CITIES.\n");
            fclose(fp);
            return 1;
        }
        // Asumiendo que los índices en el archivo son 1-based y los queremos 0-based
        // O si ya son 0-based, ajustar el índice leído si es necesario
        data.x[data.num_cities] = x;
        data.y[data.num_cities] = y;
        data.num_cities++;
    }
    fclose(fp);

    if (data.num_cities == 0)
    {
        printf("No se leyeron ciudades del archivo.\n");
        return 1;
    }

    // Ejecutar la versión CUDA
    ant_system_cuda(&data);

    return 0;
}