#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char const *argv[]){

    int tamanios[] = {32, 100, 317, 1000, 3163};

    int num_tamanios = sizeof(tamanios) / sizeof(tamanios[0]);

    // Reservar memoria dinámicamente
    int **matrizA = (int **)malloc(tamanios[4] * sizeof(int *));
    for (int i = 0; i < tamanios[4]; i++) {
        matrizA[i] = (int *)malloc(tamanios[4] * sizeof(int));
    }

    int * matrizB = (int *)malloc(tamanios[4] * sizeof(int));

    printf("Memoria reservada\n");

    // Inicializar matriz A
    //srand(time(NULL));
    for (int i = 0; i < tamanios[4]; i++) {
        for (int j = 0; j < tamanios[4]; j++) {
            //matrizA[i][j] = rand() % 11;
            matrizA[i][j] = 1;
        }
        //printf("Fila %d inicializada de la matriz A\n", i);
    }

    // Inicializar matriz B
    for (int i = 0; i < tamanios[4]; i++) {
        matrizB[i] = 0;
    }

    printf("Matrices inicializadas\n");

    for (int i = 0; i < num_tamanios; i++) {
        int tamanio_actual = tamanios[i];
        clock_t inicio, final;
        double tiempo_total;

        inicio = clock();

        for (int j = 0; j < tamanio_actual; j++) {
            for (int k = 0; k < tamanio_actual; k++) {
                matrizB[j] += matrizA[j][k];
            }
        }

        final = clock();
        tiempo_total = (double)(final - inicio) / CLOCKS_PER_SEC;
        printf("Tiempo de ejecucion para tamanio %f: %f\n", pow(tamanio_actual, 2), tiempo_total);
    }

    // Liberar memoria
    for (int i = 0; i < tamanios[4]; i++) {
        free(matrizA[i]);
    }
    free(matrizA);
    free(matrizB);

    return 0;
}