#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void transponer(float *A, float *B, int N)
{
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int totalElementos = N * N;

    if (idHilo < totalElementos)
    {
        int row = idHilo / N;
        int col = idHilo % N;
        B[col * N + row] = A[row * N + col]; // Transponer
    }
}

void imprimirMatriz(float *matriz, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matriz[i * N + j]);
        }
        printf("\n");
    }
}


int main(int argc, char const *argv[])
{
    if (argc != 2) {    
        printf("Uso: %s <tamaño de la matriz>\n", argv[0]);
        return -1;
    }
    
    int N = atoi(argv[1]); // Tamaño de la matriz
    
    size_t size = N * N * sizeof(float);
    
    float *h_A;
    float *d_A;
    float *d_AT;

    
    h_A = (float *)malloc(size);
    
    if (h_A == NULL) {
        fprintf(stderr, "Error al asignar memoria en el host\n");
        return EXIT_FAILURE;
    }
    
    srand(time(NULL));
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand()) / RAND_MAX;
    }
    
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_AT, size);

    if (d_A == NULL) {
        fprintf(stderr, "Error al asignar memoria en el dispositivo\n");
        free(h_A);
        return EXIT_FAILURE;
    }

    if (N <= 5){
        printf("Matriz original:\n");
        imprimirMatriz(h_A, N);
    }

    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades, 0);
    int tamanio_bloque = propiedades.maxThreadsPerBlock;
    
    // Mandar a proporciones del numero de hilos totales que tiene el dispositivo
    int num_bloques = (N*N + tamanio_bloque - 1) / tamanio_bloque; // esto nos sirve para dividir el trabajo en bloques de hilos

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Copiar la matriz original
    cudaEventRecord(start, 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Ejecutar kernel
    transponer<<<num_bloques, tamanio_bloque>>>(d_A, d_AT, N);
    cudaDeviceSynchronize();
    
    // Copiar la matriz transpuesta de vuelta al host
    // Copiar resultado al host
    cudaMemcpy(h_A, d_AT, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float tiempo;
    cudaEventElapsedTime(&tiempo, start, stop);
    printf("Tiempo de transposición en paralelo: %f segundos\n", tiempo / 1000.0f);
    // Liberar eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Imprimir la matriz transpuesta
    if (N <= 5){
        printf("Matriz transpuesta:\n");
        imprimirMatriz(h_A, N);
    }
    
    cudaFree(d_A);
    cudaFree(d_AT);
    free(h_A);

    return EXIT_SUCCESS;
}