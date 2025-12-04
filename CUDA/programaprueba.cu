#include <stdio.h>
#include <cuda_runtime.h> // biblioteca que nos sirve para poder extraer la informacion de nuestro dispositivo

int main(){
    cudaDeviceProp propiedades; // esta es nuestra variable del tipo cudaDeviceProp que pertenece a cuda_runctime
    int idDispositivo;
    cudaGetDevice(&idDispositivo); // obtenemos el id del dispositivo
    cudaGetDeviceProperties(&propiedades, idDispositivo); // obtenemos las propiedades del dispositivo
    printf("Informacion de la GPU %d: %s\n", idDispositivo, propiedades.name); // imprimimos el nombre del dispositivo (modelo con especificaciones)
    printf("Numero maximo de procesadores multiprocesadores (SM): %d\n", propiedades.multiProcessorCount); // imprimimos el numero de multiprocesadores
    
    printf("Numero maximo de hilos por bloque: %d\n", propiedades.maxThreadsPerBlock); // imprimimos el numero maximo de hilos por bloque
    
    printf("Numero maximo de hilos por SM: %d\n", propiedades.maxThreadsPerMultiProcessor); // imprimimos el numero maximo de hilos por multiprocesador
    
    printf("Numero maximo de bloques por grid: %d\n", propiedades.maxGridSize[0]); // imprimimos el numero maximo de bloques por grid

    printf("Numero maximo de hilos por dimension de bloque (x, y, z): (%d, %d, %d)\n", propiedades.maxThreadsDim[0], propiedades.maxThreadsDim[1], propiedades.maxThreadsDim[2]); // imprimimos el numero maximo de hilos por dimension de bloque

    printf("Tamaño de la memoria global: %ld MB\n", propiedades.totalGlobalMem/(1024*1024)); // imprimimos el tamaño de la memoria global

    printf("Tamaño de la memoria compartida por bloque: %ld bytes\n", propiedades.sharedMemPerBlock); // imprimimos el tamaño de la memoria compartida por bloque

    // numero de hilos totales en la GPU
    int hilosTotales = propiedades.multiProcessorCount * propiedades.maxThreadsPerMultiProcessor; // calculamos el numero de hilos totales en la GPU
    printf("Numero de hilos totales en la GPU: %d\n", hilosTotales); // imprimimos el numero de hilos totales en la GPU

    return 0;
}