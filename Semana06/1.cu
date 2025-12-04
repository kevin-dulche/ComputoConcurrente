#include <stdio.h>

// vamos a hacer el hola mundo en CUDA

__global__ void kernel(){
    printf("Hola mundo desde bloque %d, hilo %d\n", blockIdx.x, threadIdx.x);
    __syncthreads();
    printf("Hola despues de la sincronizacion y soy el bloque %d, hilo %d\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char const *argv[])
{
    int num_bloques = 3;
    int num_hilos = 3;
    printf("Vamos a ejecutar nuestro kernel en cuda\n");
    kernel<<<num_bloques, num_hilos>>>();
    cudaDeviceSynchronize();
    return 0;
}
