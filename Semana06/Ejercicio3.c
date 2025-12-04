#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void multiplicaMatrizxVector(float * matriz, float * vector, float * resultado, int filas, int columnas) {
    for (int i = 0; i < filas; i++) {
        resultado[i] = 0;
        for (int j = 0; j < columnas; j++) {
            resultado[i] += matriz[i * columnas + j] * vector[j];
        }
    }
}


int main(int argc, char const *argv[])
{

    if (argc != 3) {
        printf("Uso: %s <filas> <columnas>\n", argv[0]);
        return 1;
    }

    int filas = atoi(argv[1]);
    int columnas = atoi(argv[2]);

    float * matriz = (float *)malloc(filas * columnas * sizeof(float));
    float * vector = (float *)malloc(columnas * sizeof(float));
    float * resultado = (float *)malloc(filas * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            matriz[i * columnas + j] = (float) rand() / RAND_MAX;
        }
        vector[i] = (float) rand() / RAND_MAX;
    }

    printf("La matriz es:\n");
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            printf("%.2f ", matriz[i * columnas + j]);
        }
        printf("\n");
    }

    printf("El vector es:\n");
    for (int i = 0; i < columnas; i++) {
        printf("%.2f ", vector[i]);
    }
    printf("\n");

    
    multiplicaMatrizxVector(matriz, vector, resultado, filas, columnas);

    printf("El vector resultante es:\n");
    for (int i = 0; i < filas; i++) {
        printf("%.2f ", resultado[i]);
    }
    printf("\n");

    return 0;
}