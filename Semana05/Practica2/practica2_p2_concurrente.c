#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#define NUM_HILOS 5

int **matrizA;
int *matrizB;
int tamanio_actual;

// Estructura para pasar argumentos a los hilos
typedef struct {
    int inicio;
    int fin;
} Rango;

/**
 * @brief Suma los elementos de las filas de la matriz A y los guarda en la matriz B.
 * @param arg Estructura con los rangos de las filas a sumar.
 * @return void* NULL al finalizar el hilo.
 */
void *sumar_filas(void *arg) {
    Rango *rango = (Rango *)arg;
    for (int j = rango->inicio; j < rango->fin; j++) {
        for (int k = 0; k < tamanio_actual; k++) {
            matrizB[j] += matrizA[j][k];
        }
    }
    pthread_exit(NULL);
}

/**
 * @brief Función principal.
 * @return int 0 si el programa se ejecuta correctamente.
 */
int main() {
    int tamanios[] = {32, 100, 317, 1000, 3163};
    int num_tamanios = sizeof(tamanios) / sizeof(tamanios[0]);

    for (int i = 0; i < num_tamanios; i++) {
        tamanio_actual = tamanios[i];
        
        // Reservar memoria dinámicamente
        matrizA = (int **)malloc(tamanio_actual * sizeof(int *));
        for (int j = 0; j < tamanio_actual; j++) {
            matrizA[j] = (int *)malloc(tamanio_actual * sizeof(int));
        }
        matrizB = (int *)malloc(tamanio_actual * sizeof(int));

        // Inicializar matriz A y B
        for (int j = 0; j < tamanio_actual; j++) {
            for (int k = 0; k < tamanio_actual; k++) {
                matrizA[j][k] = 1;
            }
            matrizB[j] = 0;
        }

        pthread_t hilos[NUM_HILOS];
        Rango rangos[NUM_HILOS];
        int tam_parte = tamanio_actual / NUM_HILOS;

        clock_t inicio = clock();
        
        // Crear hilos
        for (int h = 0; h < NUM_HILOS; h++) {
            rangos[h].inicio = h * tam_parte;
            rangos[h].fin = (h == NUM_HILOS - 1) ? tamanio_actual : (h + 1) * tam_parte;
            pthread_create(&hilos[h], NULL, sumar_filas, (void *)&rangos[h]);
        }
        
        // Esperar a los hilos
        for (int h = 0; h < NUM_HILOS; h++) {
            pthread_join(hilos[h], NULL);
        }
        
        clock_t final = clock();
        double tiempo_total = (double)(final - inicio) / CLOCKS_PER_SEC;
        printf("Tiempo de ejecucion para tamanio %f: %f\n", pow(tamanio_actual, 2), tiempo_total);
        
        // Liberar memoria
        for (int j = 0; j < tamanio_actual; j++) {
            free(matrizA[j]);
        }
        
        free(matrizA);
        free(matrizB);

    }
    return 0;
}