#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]){

    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // nosotros tenemos el numero de procesos
    // necesitamos saber si ese valor (2^Nn) cumple que n sea mayor o igual a 2

    int N = (int)log2(size);
    // printf ("Valor de N: %d\n", N);
    // printf ("Valor de size: %d\n", size);
    if (N >= 2 && (size & (size - 1)) == 0) { // con esto garantizamos que tengamos un numero de proceso "par" y ademas N sea >= 2 (haya al menos 4 procesos)
        char nombre[MPI_MAX_PROCESSOR_NAME];
        int longitudNombre;
        MPI_Get_processor_name(nombre, &longitudNombre);
        
        if (rank % 2 == 0){// es par, recibe
            //MPI_Recv(nombre, longitudNombre, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(nombre, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Soy el proceso %d y recibi del proceso %d el mensaje %s\n", rank, rank + 1, nombre);
        } else { // es impar, envia
            //MPI_Send(nombre, longitudNombre, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Send(nombre, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
            printf("Soy el proceso %d y envie al proceso %d el mensaje %s\n", rank, rank - 1, nombre);
        }

    } else {
        printf("El numero de procesos no es potencia de 2 o no es mayor o igual a 2\n");
    }

    MPI_Finalize();

    return 0;
}