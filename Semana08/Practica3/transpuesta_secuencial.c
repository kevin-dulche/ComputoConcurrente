#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void transponerSecuencial(float *A, float *AT, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            AT[j * N + i] = A[i * N + j];
        }
    }
}


void imprimirMatriz(float *matriz, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matriz[i * N + j]);
        }
        printf("\n");
    }
}


void asignarMemoria(float **matriz, int N) {
    *matriz = (float *)malloc(N * N * sizeof(float));
    if (*matriz == NULL) {
        printf("Error al asignar memoria para la matriz.\n");
        exit(EXIT_FAILURE);
    }
}


void inicializarMatriz(float *matriz, int N) {
    srand(time(NULL));
    // Inicializar la matriz A con valores aleatorios
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matriz[i * N + j] = (float)(rand()) / RAND_MAX; 
        }
    }
}


int main(int argc, char const *argv[])
{
    if (argc != 2) {
        printf("Uso: %s <tamaño de la matriz>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("El tamaño de la matriz debe ser un número entero positivo.\n");
        return EXIT_FAILURE;
    }

    float *A, *AT;
    // Asignar memoria para la matriz A
    asignarMemoria(&A, N);
    // Asignar memoria para la matriz transpuesta
    asignarMemoria(&AT, N);

    // Inicializar la matriz A con valores aleatorios
    inicializarMatriz(A, N);

    if (N <= 5) {
        printf("Matriz A:\n");
        imprimirMatriz(A, N);
    }

    clock_t start, end;
    
    double cpu_time_used;
    
    start = clock();
    transponerSecuencial(A, AT, N);
    end = clock();
    
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución de la transposición: %f segundos\n", cpu_time_used);
    
    if (N <= 5) {
        printf("Matriz A (transpuesta):\n");
        imprimirMatriz(A, N);
    }

    // Liberar memoria
    free(A);
    free(AT);
    
    return EXIT_SUCCESS;
}