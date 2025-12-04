#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define NUM_HILOS 10  // Número de hilos a usar

typedef struct {
    int *vectorA;
    int *vectorB;
    int inicio;
    int fin;
    int suma_parcial; // Suma parcial de cada hilo
    int id;
} datos_hilos;

pthread_mutex_t mutex;
int producto_punto_global = 0;

/**
 * @brief Calcula el producto punto de un rango de elementos de dos vectores.
 * @param arg Estructura con los datos necesarios para calcular el producto punto.
 * @return void* NULL al finalizar el hilo.
 */
void *calcular_parcial(void *arg) {
    datos_hilos *data = (datos_hilos *)arg;
    int suma_local = 0;

    for (int i = data->inicio; i < data->fin; i++) {
        suma_local += data->vectorA[i] * data->vectorB[i];
        printf("Producto punto del hilo %d: [%d, %d] = %d\n", data->id, data->vectorA[i], data->vectorB[i], suma_local);
    }

    // Sección crítica: actualizar la variable global con mutex
    pthread_mutex_lock(&mutex);
    producto_punto_global += suma_local;
    printf("Producto punto global: %d del hilo %d \n", producto_punto_global, data->id);
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

/**
 * @brief Inicializa dos vectores de enteros con valores aleatorios.
 * @param vectorA Primer vector de enteros.
 * @param vectorB Segundo vector de enteros.
 * @param tamanio Tamaño de los vectores.
 * @return void
 */
void inicializar_vectores(int **vectorA, int **vectorB, int tamanio) {
    *vectorA = (int *)malloc(tamanio * sizeof(int));
    *vectorB = (int *)malloc(tamanio * sizeof(int));

    if (*vectorA == NULL || *vectorB == NULL) {
        perror("Error al asignar memoria\n");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    for (int i = 0; i < tamanio; i++) {
        (*vectorA)[i] = rand() % 11;
        (*vectorB)[i] = rand() % 11;
    }
}

/**
 * @brief Calcula el producto punto de dos vectores de enteros.
 * @param vectorA Primer vector de enteros.
 * @param vectorB Segundo vector de enteros.
 * @param tamanios Arreglo con los tamaños de los vectores.
 * @param archivo Archivo donde se escribirán los tiempos de ejecución.
 * @return void
 */
void producto_punto(int *vectorA, int *vectorB, int tamanios[], FILE *archivo) {
    clock_t inicio, final;
    double tiempo_total;

    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < 5; i++) {
        int tamanio_actual = tamanios[i];
        pthread_t hilos[NUM_HILOS];
        datos_hilos datos[NUM_HILOS];

        producto_punto_global = 0; // Resetear resultado global

        int bloque = tamanio_actual / NUM_HILOS;  // Tamaño del bloque de cada hilo

        // Inicializar los datos de cada hilo para el cálculo parcial
        for (int j = 0; j < NUM_HILOS; j++) {
            datos[j].vectorA = vectorA;
            datos[j].vectorB = vectorB;
            datos[j].inicio = j * bloque;
            datos[j].fin = (j == NUM_HILOS - 1) ? tamanio_actual : (j + 1) * bloque;
            datos[j].id = j;
        }

        inicio = clock();

        for (int j = 0; j < NUM_HILOS; j++) {
            pthread_create(&hilos[j], NULL, calcular_parcial, (void *)&datos[j]);
        }

        for (int j = 0; j < NUM_HILOS; j++) {
            pthread_join(hilos[j], NULL);
        }

        final = clock();
        tiempo_total = (double)(final - inicio) / CLOCKS_PER_SEC;
        fprintf(archivo, "%d,%f\n", tamanio_actual, tiempo_total);
        printf("Tiempo de ejecucion para tamanio %d: %f\n", tamanio_actual, tiempo_total);
    }

    pthread_mutex_destroy(&mutex);
}

/**
 * @brief Función principal.
 * @return int 0 si el programa se ejecuta correctamente.
 */
int main() {
    int tamanios[] = {10, 15, 20, 25, 30};
    int *vectorA, *vectorB;

    inicializar_vectores(&vectorA, &vectorB, tamanios[4]);

    printf("Vector A: ");
    for (int i = 0; i < tamanios[4]; i++) {
        printf("%d ", vectorA[i]);
    }

    printf("\nVector B: ");
    for (int i = 0; i < tamanios[4]; i++) {
        printf("%d ", vectorB[i]);
    }
    printf("\n");

    FILE *archivo = fopen("tiempos_concurrente.csv", "w");
    if (archivo == NULL) {
        fprintf(stderr, "Error al abrir el archivo\n");
        return EXIT_FAILURE;
    }
    fprintf(archivo, "tamanio,tiempo\n");

    producto_punto(vectorA, vectorB, tamanios, archivo);

    fclose(archivo);
    free(vectorA);
    free(vectorB);

    return 0;
}
