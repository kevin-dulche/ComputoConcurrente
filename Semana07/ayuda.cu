#include <stdio.h>

__global__ void kernel(){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hilo ID: %d\n", threadId);
}

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Nombre del dispositivo: %s\n", prop.name);
    // printf("Memoria global: %lu\n", prop.totalGlobalMem);
    // printf("Memoria compartida por bloque: %lu\n", prop.sharedMemPerBlock);
    // printf("Número de hilos por bloque: %d\n", prop.maxThreadsPerBlock);

    // prop.maxThreadsDim[0]
    printf("Máximo de hilos por bloque en x: %d\n", prop.maxThreadsDim[0]);
    printf("Máximo de hilos por bloque en y: %d\n", prop.maxThreadsDim[1]);
    printf("Máximo de hilos por bloque en z: %d\n", prop.maxThreadsDim[2]);

    dim3 gridSize(2, 1, 1); //malla de 2 bloques en x
    dim3 blockSize(3, 1, 1); //bloques de 3 hilos en x
    kernel<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    return 0;
}