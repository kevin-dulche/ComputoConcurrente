#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){

    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank % 2 == 0) {
        printf("Hola desde un proceso par %d de %d\n", rank, size);
    } else {
        printf("Hola desde un proceso impar %d de %d\n", rank, size);
    }

    MPI_Finalize();
    return 0;
}