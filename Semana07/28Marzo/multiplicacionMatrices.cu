// Multiplicacion de matrices en CUDA

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Estructura para pasar argumentos a los hilos
typedef struct {
    int id;
    int N;
    int *A;
    int *B;
    int *C;
    int inicioFila;
    int finFila;
} DatosHilo;


/**
 * @brief Multiplicación de matrices en CUDA de forma paralela.
 * @param A Matriz A.
 * @param B Matriz B.
 * @param C Matriz C.
 * @param N Tamaño de las matrices.
 * @return void.
 * @author Kevin Dulche
 */
__global__ void multiplicarParalelo(int *A, int *B, int *C, int N){
    int columna = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < N && columna < N){
        int sum = 0;
        for (int i = 0; i < N; i++){
            sum += A[fila * N + i] * B[i * N + columna];
        }
        C[fila * N + columna] = sum;
    }
}


/**
 * @brief Multiplicación de matrices de forma concurrente.
 * @param arg Argumentos del hilo (DatosHilo).
 * @return void.
 * @autor Kevin Dulche
 */
void * multiplicarConcurrente(void * arg) {
    DatosHilo *datos = (DatosHilo *)arg;
    int N = datos->N;
    
    for (int i = datos->inicioFila; i < datos->finFila; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += datos->A[i * N + k] * datos->B[k * N + j];
            }
            datos->C[i * N + j] = sum;
        }
    }
    pthread_exit(NULL);
}


/**
 * @brief Multiplicación de matrices de forma secuencial.
 * @param A Matriz A.
 * @param B Matriz B.
 * @param C Matriz C.
 * @param N Tamaño de las matrices.
 * @return void.
 * @autor Kevin Dulche
 */
void multiplicarSecuencial(int *A, int *B, int *C, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            int sum = 0;
            for (int k = 0; k < N; k++){
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


void imprimirMatriz(int *matriz, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("%d ", matriz[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Función principal.
 * @param argc Número de argumentos.
 * @param argv Argumentos.
 * @return 0 si todo sale bien, 1 si hay un error en los argumentos de entrada.
 * @autor Kevin Dulche
 */
int main(int argc, char const *argv[]){
    
    if (argc != 2)
    {
        printf("Uso: %s <tamaño de la matriz>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]); // Tamaño de la matriz

    if (N <= 0)
    {
        printf("El tamaño de la matriz debe ser un número positivo\n");
        return EXIT_FAILURE;
    }

    int *h_A, *h_B, *h_C; // Matrices en el host
    int *d_A, *d_B, *d_C; // Matrices en el dispositivo

    int *A_secuencial, *B_secuencial, *C_secuencial; // Matrices para la multiplicación secuencial
    // int *A_concurrente, *B_concurrente, *C_concurrente; // Matrices para la multiplicación concurrente

    int size = N * N * sizeof(int);

    // // Reservamos memoria en el host
    h_A = (int *)malloc(size);
    h_B = (int *)malloc(size);
    h_C = (int *)malloc(size);

    // // Reservamos memoria para las matrices secuenciales
    A_secuencial = (int *)malloc(size);
    B_secuencial = (int *)malloc(size);
    C_secuencial = (int *)malloc(size);

    // // Reservamos memoria para las matrices concurrentes
    // A_concurrente = (int *)malloc(size);
    // B_concurrente = (int *)malloc(size);
    // C_concurrente = (int *)malloc(size);

    // // Reservamos memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    srand(time(NULL)); // Semilla para números aleatorios
    for (int i = 0; i < N * N; i++){
        h_A[i] = rand() % 9; // Números aleatorios entre 0 y 9
        h_B[i] = rand() % 9; // Números aleatorios entre 0 y 9
        A_secuencial[i] = h_A[i];
        B_secuencial[i] = h_B[i];
        // A_concurrente[i] = h_A[i];
        // B_concurrente[i] = h_B[i];'
        // A_secuencial[i] = rand() % 9; // Números aleatorios entre 0 y 9
        // B_secuencial[i] = rand() % 9; // Números aleatorios entre 0 y 9
    }

    // // * Multiplicación de matrices en GPU de forma paralela.

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // Copiamos la matriz A al dispositivo
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); // Copiamos la matriz B al dispositivo

    // // Paso 1: Obtención de las propiedades del dispositivo CUDA.
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    int numHilosPorBloque[5] = {64, 128, 256, 512, 1024};

    float sumaTiempo;
    for (int i = 0; i < 5; i++)
    {
        sumaTiempo = 0;
        int num_hilos = numHilosPorBloque[i];
        for (int j = 0; j < 5; j++)
        {
            // printf("Número de hilos por bloque: %d\n", num_hilos);
            // printf("Número de bloques: %d\n", numbloques);

            // Paso 2: Cálculo del tamaño óptimo del bloque.
            int tambloque = num_hilos; // Número máximo de hilos por bloque.

            // Paso 3: Cálculo del número de bloques necesarios.
            int numbloques = (N + tambloque - 1) / tambloque; // Redondeo hacia arriba.
            printf("Número de bloques: %d\n", numbloques);

            // Paso 5: Definición del tamaño de la malla y el bloque.
            dim3 tamanoBloque(tambloque, tambloque); // Esto hace que el bloque sea bidimensional.
            dim3 tamanoMalla((N + numbloques - 1) / numbloques, (N + numbloques - 1) / numbloques); // Esto hace que la malla sea bidimensional.

            // Una malla es una colección de bloques.
            // Un bloque es una colección de hilos. 

            // Ejecutamos el kernel para paralelizar la multiplicación de matrices.
            clock_t start_paralelo = clock();
            multiplicarParalelo<<<tamanoMalla, tamanoBloque>>>(d_A, d_B, d_C, N); // Parametros: <<<malla, bloque>>>(matrizA, matrizB, matrizC, N)
            cudaDeviceSynchronize();
            clock_t end_paralelo = clock();

            double s = ((double) (end_paralelo - start_paralelo)) / CLOCKS_PER_SEC;

            cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // Copiamos la matriz C al host

            printf("Tiempo de ejecución en GPU: %f segundos para %d x %d\n", s, N, N);
            sumaTiempo += s;
        }
        printf("El tiempo promedio de ejecución es de: %f segundos con %d hilos\n", sumaTiempo / 5, num_hilos);
    }
    // * Multiplicación de matrices en CPU de forma concurrente.
    // int NUM_HILOS = sysconf(_SC_NPROCESSORS_ONLN); // Número de hilos en el sistema
    // int NUM_HILOS = 10;
    
    // int NUM_HILOS = (N < 10) ? N : 10; // Definimos el número de hilos a utilizar en función del tamaño de la matriz (máximo 10 hilos)
    // pthread_t hilos[NUM_HILOS];
    // DatosHilo datos[NUM_HILOS];

    // // Definir las filas que cada hilo procesará
    // int filasPorHilo = N / NUM_HILOS; // Número de filas que procesará cada hilo
    // int filasRestantes = N % NUM_HILOS; // Número de filas que sobran

    // struct timeval start, end;
    
    // // Usaremos gettimeofday para medir el tiempo de ejecución de los hilos ya que clock() no funciona bien con hilos
    // gettimeofday(&start, NULL);  // Inicio después de crear hilos

    // //clock_t start_concurrente = clock();

    // for (int i = 0; i < NUM_HILOS; i++) { 
    //     datos[i].id = i;
    //     datos[i].N = N;
    //     datos[i].A = A_concurrente;
    //     datos[i].B = B_concurrente;
    //     datos[i].C = C_concurrente;
    //     datos[i].inicioFila = i * filasPorHilo;
    //     datos[i].finFila = datos[i].inicioFila + filasPorHilo;

    //     if (i == NUM_HILOS - 1) { 
    //         datos[i].finFila += filasRestantes;  // Último hilo procesa filas extras
    //     }

    //     pthread_create(&hilos[i], NULL, multiplicarConcurrente, (void *)&datos[i]); // Creamos el hilo
    // }

    // // Esperamos a que terminen todos los hilos
    // for (int i = 0; i < NUM_HILOS; i++) {
    //     pthread_join(hilos[i], NULL); // Esperamos a que termine el hilo
    // }

    // // clock_t end_concurrente = clock();
    // // double tiempo_concurrente = ((double)(end_concurrente - start_concurrente)) / CLOCKS_PER_SEC;

    // // printf("Tiempo de ejecución en CPU con %d hilos: %f segundos para %d x %d\n", NUM_HILOS, tiempo_concurrente, N, N);

    // gettimeofday(&end, NULL);  // Fin después de que terminen los hilos

    // double tiempo_concurrente = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    // printf("Tiempo de ejecución en CPU con %d hilos: %f segundos para %d x %d\n", NUM_HILOS, tiempo_concurrente, N, N);


    // * Multiplicación de matrices en CPU de forma secuencial.

    double sumaTiempoSecuencial = 0;
    for (int i = 0; i < 5; i++){
        clock_t start_secuencial = clock();
        multiplicarSecuencial(A_secuencial, B_secuencial, C_secuencial, N);
        clock_t end_secuencial = clock();
        double tiempo_secuencial = ((double)(end_secuencial - start_secuencial)) / CLOCKS_PER_SEC;
        printf("Tiempo de ejecución en CPU secuencial: %f segundos para %d x %d\n", tiempo_secuencial, N, N);
        sumaTiempoSecuencial += tiempo_secuencial;
    }
    printf("El tiempo promedio de ejecución en CPU secuencial es de: %f segundos\n", sumaTiempoSecuencial / 5);

    // Imprimimos las matrices resultantes si el tamaño de la matriz es menor o igual a 5
    // if (N <= 5){
    //     printf("Matrices resultantes con GPU:\n");
    //     printf("Matriz A:\n");
    //     imprimirMatriz(h_A, N);


    //     printf("Matriz B:\n");  
    //     imprimirMatriz(h_B, N);


    //     printf("Matriz C:\n");
    //     imprimirMatriz(h_C, N);

    //     printf("Matrices resultantes con CPU concurrente:\n");
    //     printf("Matriz A:\n");
    //     imprimirMatriz(A_concurrente, N);

    //     printf("Matriz B:\n");
    //     imprimirMatriz(B_concurrente, N);

    //     printf("Matriz C:\n");
    //     imprimirMatriz(C_concurrente, N);

    //     printf("Matrices resultantes con CPU secuencial:\n");
    //     printf("Matriz A:\n");
    //     imprimirMatriz(A_secuencial, N);

    //     printf("Matriz B:\n");
    //     imprimirMatriz(B_secuencial, N);

    //     printf("Matriz C:\n");
    //     imprimirMatriz(C_secuencial, N);
    // }

    // liberamos memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    free(A_secuencial);
    free(B_secuencial);
    free(C_secuencial);

    // free(A_concurrente);
    // free(B_concurrente);
    // free(C_concurrente);

    return EXIT_SUCCESS;
}