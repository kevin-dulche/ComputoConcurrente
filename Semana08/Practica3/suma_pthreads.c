#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

typedef struct {
    float *A;
    float *B;
    float *C;
    int inicio;
    int fin;
    int N;
    int id;
} parametros;

void *suma_matrices(void *arg) {
    parametros *args = (parametros *)arg;
    for (int i = args->inicio; i < args->fin; ++i) {
        for (int j = 0; j < args->N; ++j) {
            int idx = i * args->N + j;
            args->C[idx] = args->A[idx] + args->B[idx];
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <tamaño de la matriz (N)> <número de hilos>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    int N = atoi(argv[1]); // Matriz N x N
    int num_hilos = atoi(argv[2]);
    int filas_por_hilo = N / num_hilos;
    int sobrante = N % num_hilos;

    struct timeval inicio, final;

    // Reservamos espacio para las matrices
    float *A = (float *)malloc(sizeof(float) * N * N);
    float *B = (float *)malloc(sizeof(float) * N * N);
    float *C = (float *)malloc(sizeof(float) * N * N);

    // Inicializamos A y B con valores aleatorios
    for (int i = 0; i < N * N; ++i) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    parametros args[num_hilos];
    pthread_t hilos[num_hilos];

    gettimeofday(&inicio, NULL);

    for (int i = 0; i < num_hilos; ++i) {
        int inicio = i * filas_por_hilo;
        int fin = (i + 1) * filas_por_hilo;

        if (i == num_hilos - 1)
            fin += sobrante;

        args[i].A = A;
        args[i].B = B;
        args[i].C = C;
        args[i].inicio = inicio;
        args[i].fin = fin;
        args[i].N = N;
        args[i].id = i;

        pthread_create(&hilos[i], NULL, suma_matrices, (void *)&args[i]);
    }

    for (int i = 0; i < num_hilos; ++i)
        pthread_join(hilos[i], NULL);

    gettimeofday(&final, NULL);
    double tiempo = (final.tv_sec - inicio.tv_sec) + (final.tv_usec - inicio.tv_usec) / 1000000.0;
    printf("El tiempo de ejecución es: %f segundos\n", tiempo);

    free(A);
    free(B);
    free(C);
    return 0;
}
