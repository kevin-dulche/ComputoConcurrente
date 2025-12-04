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

    // Vamos a asumir que nuestros bloques van a ser de 4 elementos
    int columnas = 4;
    int fila_id = rank/columnas;

    MPI_Comm comunicador_fila; // MPI_Comm es un tipo de dato que representa un comunicador 'local'

    MPI_Comm_split(MPI_COMM_WORLD, fila_id, rank, &comunicador_fila); // MPI_Comm_split(MPI_COMM_WORLD, filas, rank, &comunicador_fila);
    // MPI_COMM_WORLD es el comunicador global, filas es el color del nuevo comunicador, rank es el rango del proceso en el comunicador original y &comunicador_fila es el nuevo comunicador

    int rank_local, size_local; // rank_local es el id del proceso en el nuevo comunicador y size_local es el tamaño del nuevo comunicador
    MPI_Comm_rank(comunicador_fila, &rank_local); // obtiene el rank del proceso en el nuevo comunicador
    MPI_Comm_size(comunicador_fila, &size_local); // obtiene el tamaño del nuevo comunicador

    //printf("Soy el proceso %d de %d y me estoy ejecutando en el host %d\n", rank_local, size_local, fila_id);

    // nos piden que los procesos calculen la suma de sus ranks
    // y que el 0 imprima ese valor
    // reduce en MPI nos permite ejecutar funciones en los procesos y que se envie al resultado a un proceso especifico
    // para nuestro caso, el reduce va a ir al proceso 0 (local)
    int valor_local = rank_local; // cada proceso va a tener como valor su ranking local
    int suma;
    MPI_Reduce(&valor_local, // valor que se va a sumar por cada proceso
                &suma, // valor que vamos a guardar
                1,  // guardamos un elemnto (tamaño de suma)
                MPI_INT,  // tipo de dato que vamos a trabajar es del tipo entero MPI
                MPI_SUM,  // tipo de operacion de reduccion que vamos a hacer, en este caso la suma de los valores locales
                0, // solo el proceso 0 va a imprimir el resultado
                comunicador_fila); // nuestro comunicador de procesos

    if (rank_local == 0){ // solo el proceso 0 va a imprimir el resultado
        printf("La suma de los IDs de mis amigos de la fila %d es %d\n", fila_id, suma);
    }
    
    // Finalización del entorno MPI
    MPI_Finalize();
    return 0;
}