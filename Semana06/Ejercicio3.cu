#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


__global__ void  multiplicaMatrizxVector(float * matriz, float * vector, float * resultado, int filas, int columnas) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idHilo < filas) {
        printf("idHilo: %d\n", idHilo);
        printf("threadIdx.x: %d\n", threadIdx.x);
        printf("blockIdx.x: %d\n", blockIdx.x);
        printf("blockDim.x: %d\n", blockDim.x);
        printf("idHilo: %d calculando...\n", idHilo);
        resultado[idHilo] = 0.0;
        for (int i = 0; i < columnas; i++) {
            resultado[idHilo] += matriz[idHilo * columnas + i] * vector[i];
        }
    }
}


int main(int argc, char const *argv[])
{

    if (argc != 3) {
        printf("Uso: %s <filas> <columnas>\n", argv[0]);
        return 1;
    }

    int filas = atoi(argv[1]);
    int columnas = atoi(argv[2]);

    float * matriz_h = (float *)malloc(filas * columnas * sizeof(float));
    float * vector_h = (float *)malloc(columnas * sizeof(float));
    float * resultado_h = (float *)malloc(filas * sizeof(float));

    float * matriz_d, * vector_d, * resultado_d;

    cudaMalloc(&matriz_d, filas * columnas * sizeof(float));
    cudaMalloc(&vector_d, columnas * sizeof(float));
    cudaMalloc(&resultado_d, filas * sizeof(float));
    
    srand(time(NULL));
    
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            matriz_h[i * columnas + j] = (float) rand() / RAND_MAX;
        }
        vector_h[i] = (float) rand() / RAND_MAX;
    }
    
    cudaMemcpy(matriz_d, matriz_h, filas * columnas * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vector_d, vector_h, columnas * sizeof(float), cudaMemcpyHostToDevice);

    printf("La matriz es:\n");
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            printf("%.2f ", matriz_h[i * columnas + j]);
        }
        printf("\n");
    }

    printf("\nEl vector es:\n");
    for (int i = 0; i < columnas; i++) {
        printf("%.2f ", vector_h[i]);
    }
    printf("\n\n");

    // Sacamos las propiedades del dispositivo
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades, 0);
    int tamanio_bloque = propiedades.maxThreadsPerBlock;
    
    // Mandar a proporciones del numero de hilos totales que tiene el dispositivo
    int num_bloques = (filas + tamanio_bloque - 1) / tamanio_bloque; // esto nos sirve para dividir el trabajo en bloques de hilos

    multiplicaMatrizxVector<<<num_bloques, tamanio_bloque>>>(matriz_d, vector_d, resultado_d, filas, columnas);
    cudaDeviceSynchronize();

    cudaMemcpy(resultado_h, resultado_d, filas * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nEl vector resultante es:\n");
    for (int i = 0; i < filas; i++) {
        printf("%.2f ", resultado_h[i]);
    }
    printf("\n");

    free(matriz_h);
    free(vector_h);
    free(resultado_h);

    cudaFree(matriz_d);
    cudaFree(vector_d);
    cudaFree(resultado_d);

    return 0;
}