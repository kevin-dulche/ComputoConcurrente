#include <stdio.h>

__global__ void kernel(){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hilo ID: %d\n", threadId);
}

int main(){
    dim3 gridSize(2, 1, 1); //malla de 2 bloques en x
    dim3 blockSize(3, 1, 1); //bloques de 3 hilos en x
    kernel<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    return 0;
}