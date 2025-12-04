# include <stdio.h>
# include <stdlib.h>
# include <time.h>

int main(int argc, char const *argv[])
{

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int N = 1000000000;
    int tambloque = prop.maxThreadsPerBlock;
    int numbloques = (N + tambloque - 1)/tambloque;

    printf("Número de bloques: %d\n", numbloques);
    printf("Numero de hilos por bloque: %d\n", tambloque);
    printf("Numero de hilos totales: %d\n", numbloques * tambloque);

    return 0;
}
