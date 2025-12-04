#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

int main(int argc, char const *argv[]){
    // Imprimir cuantos hilos puedo crear
    int num_hilos = sysconf(_SC_NPROCESSORS_ONLN);
    printf("Numero maximo de hilos: %d\n", num_hilos);

    int N = 1025;
    int tambloque = 1024;

    int numbloques = (N + tambloque - 1) / tambloque; 
    printf("Número de bloques: %d\n", numbloques);
    return 0;
}
