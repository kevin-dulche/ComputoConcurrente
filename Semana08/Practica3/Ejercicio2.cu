// Transposicion de una matriz N x N de tipo float

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

// Estructura para pasar argumentos a los hilos
typedef struct {
    int id;
    int N;
    float *A;
    float *B;
    int inicioFila;
    int finFila;
} DatosHilo;

// Implementacion Secuencial

__host__ void transponerSecuencial(float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

// Implementacion Concurrente

__host__ void * transponerConcurrente(void *arg) {
    DatosHilo *datos = (DatosHilo *)arg;
    int N = datos->N;
    float *A = datos->A;
    float *B = datos->B;
    int inicioFila = datos->inicioFila;
    int finFila = datos->finFila;
    for (int i = inicioFila; i < finFila; i++) {
        for (int j = 0; j < N; j++) {
            B[j * N + i] = A[i * N + j];
        }
    }
    return NULL;
}

// Implementacion Paralela

__global__ void transponerParalelo(float *A, float *B, int N) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    int columna = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < N && columna < N) {
        B[columna * N + fila] = A[fila * N + columna];
    }
}

// Imprimir matriz
__host__ void imprimirMatriz(float *matriz, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matriz[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char const *argv[])
{
    int N = 8192; // Tamaño de la matriz
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_B_Paralelo;
    float *d_A, *d_B;
    cudaEvent_t start, stop;
    float tiempoSecuencial, tiempoParalelo;
    
    // Asignar memoria en el host

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_B_Paralelo = (float *)malloc(size);

    srand(time(NULL));

    // Inicializar la matriz A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = float(rand()) / RAND_MAX; 
        }
    }
    // Imprimir matriz original
    if (N <= 5) {
        printf("Matriz original:\n");
        imprimirMatriz(h_A, N);
    }

    clock_t startSecuencial, endSecuencial;
    startSecuencial = clock();
    transponerSecuencial(h_A, h_B, N); // Transponer de forma secuencial
    endSecuencial = clock();
    tiempoSecuencial = ((double)(endSecuencial - startSecuencial)) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución en CPU secuencial: \t %f segundos para %d x %d\n", tiempoSecuencial, N, N);

    // Imprimir matriz transpuesta
    if (N <= 5) {
        printf("Matriz transpuesta (secuencial):\n");
        imprimirMatriz(h_B, N);
    }

    int numHilos = 10;
    pthread_t hilos[numHilos];
    DatosHilo datos[numHilos];
    int filasPorHilo = N / numHilos;
    int filasRestantes = N % numHilos;

    struct timeval startConcurrente, endConcurrente;
    gettimeofday(&startConcurrente, NULL);

    // Crear hilos para la transposición concurrente
    for (int i = 0; i < numHilos; i++) {
        datos[i].id = i;
        datos[i].N = N;
        datos[i].A = h_B;
        datos[i].B = h_A;
        datos[i].inicioFila = i * filasPorHilo;
        datos[i].finFila = (i + 1) * filasPorHilo;

        if (i == numHilos - 1) {
            datos[i].finFila += filasRestantes; // Asignar filas restantes al último hilo
        }

        pthread_create(&hilos[i], NULL, transponerConcurrente, (void *)&datos[i]);
    }

    // Esperar a que terminen todos los hilos
    for (int i = 0; i < numHilos; i++) {
        pthread_join(hilos[i], NULL);
    }
    gettimeofday(&endConcurrente, NULL);
    double tiempoConcurrente = (endConcurrente.tv_sec - startConcurrente.tv_sec) + (endConcurrente.tv_usec - startConcurrente.tv_usec) / 1e6;
    printf("Tiempo de ejecución en CPU concurrente:  %f segundos para %d x %d\n", tiempoConcurrente, N, N);

    // Imprimir matriz transpuesta
    if (N <= 5) {
        printf("Matriz transpuesta (concurrente):\n");
        imprimirMatriz(h_A, N);
    }

    // Asignar memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    
    // Copiar la matriz A al dispositivo
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
    
    // Definir el tamaño de bloque y grid
    // Paso 1: Obtención de las propiedades del dispositivo CUDA.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Paso 2: Cálculo del tamaño óptimo del bloque.
    int tambloque = prop.maxThreadsPerBlock; // Número máximo de hilos por bloque.

    // Paso 3: Cálculo del número de bloques necesarios.
    int numbloques = (N + tambloque - 1) / tambloque; // Redondeo hacia arriba.

    // Paso 5: Definición del tamaño de la malla y el bloque.
    dim3 tamanoBloque(numbloques, numbloques); // Esto hace que el bloque sea bidimensional.
    dim3 tamanoMalla((N + numbloques - 1) / numbloques, (N + numbloques - 1) / numbloques); // Esto hace que la malla sea bidimensional.

    // Paso 6: Lanzamiento del kernel.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    transponerParalelo<<<tamanoMalla, tamanoBloque>>>(d_A, d_B, N);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiempoParalelo, start, stop);
    printf("Tiempo de ejecución en GPU paralelo: \t %f segundos para %d x %d\n", tiempoParalelo / 1000, N, N);
    
    // Copiar la matriz transpuesta de vuelta al host
    cudaMemcpy(h_B_Paralelo, d_B, size, cudaMemcpyDeviceToHost);
    
    // Imprimir matriz transpuesta
    if (N <= 5) {
        printf("Matriz transpuesta (paralelo):\n");
        imprimirMatriz(h_B_Paralelo, N);
    }

    // Liberar memoria
    free(h_A);
    free(h_B);
    free(h_B_Paralelo);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
