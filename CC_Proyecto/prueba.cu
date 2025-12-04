#include <stdio.h>

__global__ void checkAtomicAddSupport() {
    double test = 1.0;
    atomicAdd(&test, 1.0);  // Si no hay error de compilación, la GPU lo soporta
}

int main() {
    checkAtomicAddSupport<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Si no hay error, tu GPU soporta atomicAdd para doubles.\n");
    return 0;
}