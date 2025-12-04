#include <stdio.h>
#include <time.h>
#include <stdlib.h>

/**
 * @brief Inicializa dos vectores de enteros con valores aleatorios.
 * @param vectorA Primer vector de enteros.
 * @param vectorB Segundo vector de enteros.
 * @param tamanio Tamaño de los vectores.
 * @return void
 */
void inicializar_vectores(int ** vectorA, int ** vectorB, int tamanios[]){

    *vectorA = (int *)malloc(tamanios[4] * sizeof(int));
    *vectorB = (int *)malloc(tamanios[4] * sizeof(int));

    if (*vectorA == NULL || *vectorB == NULL){
        perror("Error al asignar memoria\n");
        exit(EXIT_FAILURE);
    }

    //init random seed
    srand(time(NULL));
    for (int i = 0; i < tamanios[4]; i++){
        // llenar vectores con valores aleatorios del 0 al 10
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
void producto_punto(int * vectorA, int * vectorB, int tamanios[], FILE * archivo){
    clock_t inicio, final;
    double tiempo_total;
    int producto_punto;
    
    for (int i = 0; i < 5; i++){
        producto_punto = 0;
        inicio = clock();
        for (int j = 0; j < tamanios[i]; j++){
            producto_punto += vectorA[j] * vectorB[j];
        }
        final = clock();
        tiempo_total = (double)(final - inicio) / CLOCKS_PER_SEC;
        fprintf(archivo, "%d,%f\n", tamanios[i], tiempo_total);
        printf("Tiempo de ejecucion para tamanio %d: %f\n", tamanios[i], tiempo_total);
        //printf("Producto punto: %d\n", producto_punto);
    }
}

/**
 * @brief Función principal.
 * @return int 0 si el programa se ejecuta correctamente.
 */
int main(int argc, char const *argv[]){

    int tamanios[] = {1000, 10000, 100000, 1000000, 10000000};

    int * vectorA;
    int * vectorB;

    inicializar_vectores(&vectorA, &vectorB, tamanios);

    // crear csv de tiempos ejecucion de cada tamaño
    FILE * archivo;
    archivo = fopen("tiempos_secuencial.csv", "w");
    fprintf(archivo, "tamanio,tiempo\n");

    // producto punto
    producto_punto(vectorA, vectorB, tamanios, archivo);

    fclose(archivo);

    free(vectorA);
    free(vectorB);

    return EXIT_SUCCESS;
}