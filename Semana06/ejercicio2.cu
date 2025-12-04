#include <stdio.h>

// vamos a hacer el hola mundo en CUDA

__global__ void kernel(){
    printf("Hola mundo desde bloque %d, hilo %d\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char const *argv[])
{
    int num_bloques = 3;
    int num_hilos = 3;
    cudaDeviceProp propiedades; // estructura que contiene las propiedades del dispositivo

    cudaGetDeviceProperties(&propiedades, 0); // obtenemos las propiedades del dispositivo, por lo regular es 0 porque ]

    num_bloques = propiedades.multiProcessorCount; // numero de bloques es igual al numero de multiprocesadores
    num_hilos = propiedades.maxThreadsPerBlock; // numero de hilos es igual al numero maximo de hilos por multiprocesador

    printf("Vamos a ejecutar nuestro kernel en cuda\n");
    kernel<<<num_bloques, num_hilos>>>();
    cudaDeviceSynchronize();
    printf("En total puedo ejecutar %d hilos\n", num_bloques*num_hilos);
    return 0;
}
