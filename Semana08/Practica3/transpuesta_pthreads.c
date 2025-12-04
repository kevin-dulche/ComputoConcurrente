#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>

typedef struct {
    float *A;
    float *AT;
    int N;
    int fila;
} TranspuestaArgs;

void * transponerParcial(void *args) {
    TranspuestaArgs *tArgs = (TranspuestaArgs *)args;
    float *A = tArgs->A;
    float *AT = tArgs->AT;
    int N = tArgs->N;
    int fila = tArgs->fila;

    // Transponer la fila correspondiente
    for (int j = 0; j < N; j++) {
        AT[j * N + fila] = A[fila * N + j];
    }
    pthread_exit(NULL);
}

void imprimirMatriz(float *matriz, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matriz[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 3) {
        printf("Uso: %s <tamaño de la matriz> <numHilos>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int N = atoi(argv[1]);
    
    if (N <= 0) {
        printf("El tamaño de la matriz debe ser un número entero positivo.\n");
        return EXIT_FAILURE;
    }
    
    int numHilos = atoi(argv[2]);
    if (numHilos <= 0) {
        printf("El número de hilos debe ser un número entero positivo.\n");
        return EXIT_FAILURE;
    }

    float *A, *AT;
    // Asignar memoria para la matriz transpuesta
    AT = (float *)malloc(N * N * sizeof(float));
    if (AT == NULL) {
        printf("Error al asignar memoria para la matriz transpuesta.\n");
        return EXIT_FAILURE;
    }

    // Asignar memoria para las matrices A y B
    A = (float *)malloc(N * N * sizeof(float));
    if (A == NULL) {
        printf("Error al asignar memoria para la matriz A.\n");
        return EXIT_FAILURE;
    }
    
    // Inicializar la matriz A con valores aleatorios
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)(rand()) / RAND_MAX;
        }
    }
    
    if (N <= 5) {
        printf("Matriz A (original):\n");
        imprimirMatriz(A, N);
    }

    pthread_t hilos[numHilos];
    TranspuestaArgs args[numHilos];
    // Dividir el trabajo entre los hilos
    // Crear hilos para transponer la matriz
    struct timeval startConcurrente, endConcurrente;
    gettimeofday(&startConcurrente, NULL);
    for (int i = 0; i < numHilos; i++) {
        args[i].A = A;
        args[i].AT = AT;
        args[i].N = N;
        args[i].fila = i;
        
        pthread_create(&hilos[i], NULL, transponerParcial, (void *)&args[i]);
    }

    for (int i = 0; i < numHilos; i++) {
        pthread_join(hilos[i], NULL);
    }
    gettimeofday(&endConcurrente, NULL);
    long secondsConcurrente = endConcurrente.tv_sec - startConcurrente.tv_sec;

    double tiempoConcurrente = (endConcurrente.tv_sec - startConcurrente.tv_sec) + (endConcurrente.tv_usec - startConcurrente.tv_usec) / 1e6;
    printf("Tiempo de ejecución en CPU concurrente:  %f segundos con %d hilos\n", tiempoConcurrente, numHilos);

    if (N <= 5) {
        printf("Matriz A (transpuesta):\n");
        imprimirMatriz(A, N);
    }

    // Liberar memoria
    free(A);

    return 0;
}
