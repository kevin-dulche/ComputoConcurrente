#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>


__global__ void sumarMatricesParalela(float *A, float *B, float *C, int N) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (fila < N && columna < N) {
        int idx = fila * N + columna;
        C[idx] = A[idx] + B[idx];
    }
}


int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 10;
    srand(time(NULL));
    
    int tamanio_matriz = N * N * sizeof(float);
    
    // Reservar memoria en el host
    float *A_h = (float*)malloc(tamanio_matriz );
    float *B_h = (float*)malloc(tamanio_matriz );
    float *C_h = (float*)malloc(tamanio_matriz );
    
    // Inicializar matrices
    for (int i = 0; i < N * N; i++) {
        A_h[i] = (float)rand()/ RAND_MAX;
        B_h[i] = (float)rand()/ RAND_MAX;
    }
    
    // Reservar memoria en el dispositivo
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&B_d, tamanio_matriz);
    cudaMalloc((void **)&A_d, tamanio_matriz);
    cudaMalloc((void **)&C_d, tamanio_matriz);
    
    
    // Obtener propiedades de la GPU
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades, 0);
    
    int tamBloque = propiedades.maxThreadsPerBlock;
    int numBloques = (N + tamBloque - 1) / tamBloque;
    
    dim3 hilosPorBloque(numBloques, numBloques);
    dim3 tamanoMalla((N + numBloques - 1) / numBloques, (N + numBloques - 1) / numBloques);
    
    // Medir tiempo de ejecución
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Copiar datos al dispositivo
    cudaMemcpy(A_d, A_h, tamanio_matriz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, tamanio_matriz, cudaMemcpyHostToDevice);
    
    sumarMatricesParalela<<<tamanoMalla, hilosPorBloque>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiempo;
    cudaEventElapsedTime(&tiempo, start, stop);
    
    // Copiar resultado de vuelta al host
    cudaMemcpy(C_h, C_d, tamanio_matriz, cudaMemcpyDeviceToHost);
    
    // Mostrar tiempo de ejecución del kernel
    printf("Tiempo de ejecución del kernel en paralelo: %f segundos\n", tiempo / 1000.0f);

    if (N<= 6)
    {
        printf("\nEl vector a = \n");
        for (int i = 0; i < N * N; i++)
        {
            printf("%.2f ", A_h[i]);
            if ((i + 1) % N == 0)
                printf("\n");
        }
        printf("\nEl vector b = \n");
        for (int i = 0; i < N * N; i++)
        {
            printf("%.2f ", B_h[i]);
            if ((i + 1) % N == 0)
                printf("\n");
        }
        printf("\nEl vector c = \n");
        for (int i = 0; i < N * N; i++)
        {
            printf("%.2f ", C_h[i]);
            if ((i + 1) % N == 0)
                printf("\n");
        }
        
    }
    
    // Liberar memoria
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);
    
    return 0;
}