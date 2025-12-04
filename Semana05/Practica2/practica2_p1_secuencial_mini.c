#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char const *argv[]){

    int tamanios[] = {5, 10, 15, 20, 25};

    int * vectorA;
    int * vectorB;

    vectorA = (int *)malloc(tamanios[4] * sizeof(int));
    vectorB = (int *)malloc(tamanios[4] * sizeof(int));

    //init random seed
    srand(time(NULL));
    for (int i = 0; i < tamanios[4]; i++){
        // llenar vectores con valores aleatorios del 0 al 10
        vectorA[i] = rand() % 11;
        vectorB[i] = rand() % 11;
    }

    // crear csv de tiempos ejecucion de cada tamaño
    FILE * archivo;
    archivo = fopen("tiempos_secuencial.csv", "w");
    fprintf(archivo, "tamanio,tiempo\n");

    clock_t inicio, final;
    double tiempo_total;

    // producto punto

    int producto_punto;
    
    for (int i = 0; i < 5; i++){
        producto_punto = 0;
        inicio = clock();
        for (int j = 0; j < tamanios[i]; j++){
            producto_punto += vectorA[j] * vectorB[j];
            printf("Producto punto de %d y %d: %d\n", vectorA[j], vectorB[j], producto_punto);
        }
        final = clock();
        tiempo_total = (double)(final - inicio) / CLOCKS_PER_SEC;
        fprintf(archivo, "%d,%f\n", tamanios[i], tiempo_total);
        printf("Tiempo de ejecucion para tamanio %d: %f\n", tamanios[i], tiempo_total);
        printf("Producto punto: %d\n", producto_punto);
    }

    fclose(archivo);

    free(vectorA);
    free(vectorB);

    return 0;
}