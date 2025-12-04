#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_HILOS 10

typedef struct {
    int id;
    int N;
    int *A;
    int *B;
    int *C;
    int inicioFila;
    int finFila;
} DatosHilo;

/**
 * @brief Función que ejecutan los hilos para multiplicar una porción de la matriz.
 */
void *multiplicar(void *arg) {
    DatosHilo *datos = (DatosHilo *)arg;
    int N = datos->N;
    
    for (int i = datos->inicioFila; i < datos->finFila; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += datos->A[i * N + k] * datos->B[k * N + j];
            }
            datos->C[i * N + j] = sum;
        }
    }
    printf("Hilo %d terminó de procesar las filas %d a %d\n", datos->id, datos->inicioFila, datos->finFila - 1);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamaño de la matriz>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int *A, *B, *C;
    
    // Reservamos memoria para las matrices
    A = (int *)malloc(N * N * sizeof(int));
    B = (int *)malloc(N * N * sizeof(int));
    C = (int *)malloc(N * N * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10 + 1;
        B[i] = rand() % 10 + 1;
    }

    pthread_t hilos[NUM_HILOS];
    DatosHilo datos[NUM_HILOS];

    // Definir las filas que cada hilo procesará
    int filasPorHilo = N / NUM_HILOS;
    int filasRestantes = N % NUM_HILOS;

    clock_t start = clock();

    for (int i = 0; i < NUM_HILOS; i++) {
        datos[i].id = i;
        datos[i].N = N;
        datos[i].A = A;
        datos[i].B = B;
        datos[i].C = C;
        datos[i].inicioFila = i * filasPorHilo;
        datos[i].finFila = datos[i].inicioFila + filasPorHilo;

        if (i == NUM_HILOS - 1) { 
            datos[i].finFila += filasRestantes;  // Último hilo procesa filas extras
        }

        pthread_create(&hilos[i], NULL, multiplicar, (void *)&datos[i]);
    }

    // Esperamos a que terminen todos los hilos
    for (int i = 0; i < NUM_HILOS; i++) {
        pthread_join(hilos[i], NULL);
    }

    clock_t end = clock();
    double tiempo = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Tiempo de ejecución en CPU con %d hilos: %f segundos\n", NUM_HILOS, tiempo);

    if (N < 10) {
        printf("Matriz A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", A[i * N + j]);
            }
            printf("\n");
        }
        printf("\nMatriz B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", B[i * N + j]);
            }
            printf("\n");
        }
        printf("\nMatriz C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i * N + j]);
            }
            printf("\n");
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}