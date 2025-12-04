#include <stdio.h>
#include <mpi.h>

int main(int argc, char * argv[]){
    // Inicialización de MPI
    int rank, size;
    // rank es el id del proceso y size es el tamaño del comunicador

    MPI_Init(&argc, &argv); // (NULL, NULL)

    char nombre_proceso[MPI_MAX_PROCESSOR_NAME]; // es un arreglo de caracteres de tamaño MPI_MAX_PROCESSOR_NAME (255)

    int longitud_nombre_proceso; // longitud del nombre del proceso

    MPI_Get_processor_name(nombre_proceso, &longitud_nombre_proceso); // obtiene el nombre del proceso
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Soy el proceso %d de %d y me estoy ejecutando en el host %s\n", rank, size, nombre_proceso);
    MPI_Finalize();
    return 0;
}