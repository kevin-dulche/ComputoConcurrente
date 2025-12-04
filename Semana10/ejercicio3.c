#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){

    int identificadorProceso, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &identificadorProceso);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100;

    int base = N / size;

    int sobrante = N % size;

    int inicio;

    if (identificadorProceso < sobrante) {
        inicio = identificadorProceso * base + identificadorProceso;
    } else {
        inicio = identificadorProceso * base + sobrante;
    }
    
    if (identificadorProceso < sobrante) {
        base++;
    }

    int fin = inicio + base;

    printf("Soy el proceso %d de %d. Me tocan los índices del %d al %d (%d elementos)\n", identificadorProceso, size, inicio, fin - 1, fin - inicio);

    MPI_Finalize();
    return 0;
}