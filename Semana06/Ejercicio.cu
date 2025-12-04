#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void suma_vectores(int *A, int *B, int *C, int dimension)
{
    // Calculamos el id del hilo
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;

    if (idHilo < dimension)
    {
        C[idHilo] = A[idHilo] + B[idHilo];
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        printf("Uso: %s <dimension>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int dimension = atoi(argv[1]);

    int *A_dispositivo, *B_dispositivo, *C_dispositivo;
    int *A_host, *B_host, *C_host;

    A_host = (int*)malloc(dimension * sizeof(int));
    B_host = (int*)malloc(dimension * sizeof(int));
    C_host = (int*)malloc(dimension * sizeof(int));

    for (int i = 0; i < dimension; i++)
    {
        A_host[i] = 10 + rand() % 90;
        B_host[i] = 10 + rand() % 90;
    }

    // Reservamos memoria en el dispositivo
    cudaMalloc(&A_dispositivo, dimension * sizeof(int));
    cudaMalloc(&B_dispositivo, dimension * sizeof(int));
    cudaMalloc(&C_dispositivo, dimension * sizeof(int));

    // Copiamos la memoria del host al dispositivo
    cudaMemcpy(A_dispositivo, A_host, dimension * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dispositivo, B_host, dimension * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(C_dispositivo, C_host, dimension * sizeof(int), cudaMemcpyHostToDevice);

    // Sacamos las propiedades del dispositivo
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades, 0);
    int tamanio_bloque = propiedades.maxThreadsPerBlock;
    
    // Mandar a proporciones del numero de hilos totales que tiene el dispositivo
    int num_bloques = (dimension + tamanio_bloque - 1) / tamanio_bloque; // esto nos sirve para dividir el trabajo en bloques de hilos

    // Lanzamos el kernel
    suma_vectores<<<num_bloques, tamanio_bloque>>>(A_dispositivo, B_dispositivo, C_dispositivo, dimension);

    // Regresamos la memoria del dispositivo al host
    cudaMemcpy(C_host, C_dispositivo, dimension * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimimos los resultados
    printf("Vector A:\n");
    for (int i = 0; i < dimension; i++)
    {
        printf("%d ", A_host[i]);
    }
    printf("\n");

    printf("Vector B:\n");
    for (int i = 0; i < dimension; i++)
    {
        printf("%d ", B_host[i]);
    }
    printf("\n");

    printf("Vector C:\n");
    for (int i = 0; i < dimension; i++)
    {
        printf("%d ", C_host[i]);
    }
    printf("\n");

    // Liberamos la memoria
    free(A_host);
    free(B_host);
    free(C_host);

    cudaFree(A_dispositivo);
    cudaFree(B_dispositivo);
    cudaFree(C_dispositivo);

    return EXIT_SUCCESS;
}
