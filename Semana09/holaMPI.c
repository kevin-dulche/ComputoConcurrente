// Este programa es el hola mundo en mpi

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{   
    MPI_Init(&argc, &argv);
    
    printf("Hola mundo\n");
    
    MPI_Finalize();

    return 0;
}
