// A = 1, B = 2, C = 3, D = 4, E = 5, F = 6, G = 7, H = 8, I = 9, J = 10, K = 11, L = 12, 
// M = 13, N = 14, O = 15, P = 16, Q = 17, R = 18, S = 19, T = 20, U = 21, V = 22, W = 23, X = 24, Y = 25, Z = 26

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Transpone una matriz
 * @param A Matriz a transponer
 * @param B Matriz transpuesta
 * @param m Número de filas
 * @param n Número de columnas
 * @return void
 * @author Kevin Dulche
 */
__global__ void transponer(int *A, int *B, int m, int n) 
{
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int totalElementos = m * n;

    if (idHilo < totalElementos)
    {
        /** int row = idHilo / n;
            // ? ¿Qué hace?
            Convierte el índice lineal idHilo a la fila correspondiente.

            ¿Por qué funciona?
            Cada fila tiene n elementos.

            Si tú tienes idHilo = 7 y n = 3:

            7 / 3 = 2 → estás en la fila 2 (indexada desde 0). */
        int row = idHilo / n; // * Fila

        /** int col = idHilo % n;
            // ? ¿Qué hace?
            Convierte el índice lineal idHilo a la columna correspondiente.

            ¿Por qué funciona?
            idHilo % n da el desplazamiento dentro de la fila.

            Siguiendo el ejemplo:

            7 % 3 = 1 → columna 1 de la fila 2. */
        int col = idHilo % n; // * Columna

        /** // ? ¿Qué representa A[row * n + col]?
            // * La posición exacta en el arreglo lineal que corresponde a la celda (row, col) de la matriz 2D.

            La matriz B transpuesta tiene tamaño n x m (filas y columnas intercambiadas).
            La celda (col, row) en 2D se almacena como col * m + row en el arreglo lineal B. */
        B[col * m + row] = A[row * n + col]; // Transponer
    }
}

/**
 * @brief Función principal
 * @param argc Número de argumentos
 * @param argv Argumentos
 * @return int Estado de finalización del programa
 * @author Kevin Dulche
 */
int main(int argc, char const *argv[])
{
    /* • Calcula el tamaño de una matriz rectangular (m filas por n columnas), donde:
        –m = posición de la primera letra de tu segundo nombre (si no tienes segundo nombre,
        usa la primera letra de tu apellido materno) multiplicada por 20.
        – n = posición de la última letra de tu primer apellido multiplicada por 30.*/
    char letra_m = 'U';
    char letra_n = 'E';

    int pos_m = (letra_m >= 'a') ? letra_m - 'a' + 1 : letra_m - 'A' + 1;
    int pos_n = (letra_n >= 'a') ? letra_n - 'a' + 1 : letra_n - 'A' + 1;
    
    int m = pos_m * 20; // 21 * 20 = 420
    int n = pos_n * 30; // 5 * 30 = 150

    // m = 2; Pruebas
    // n = 3; Pruebas

    m = 10000;
    n = 10000;

    int totalElementos = m * n;

    printf("Tamaño matriz: %d x %d\n", m, n);
    
    int *h_A = (int *)malloc(totalElementos * sizeof(int)); // Memoria en CPU
    int *h_B = (int *)malloc(totalElementos * sizeof(int)); // Memoria en CPU
    
    srand(time(NULL));
    for (int i = 0; i < totalElementos; i++) {
        h_A[i] = rand() % 100 + 1;
    }
    
    // • Usa memoria dinámica en CUDA para alojar matrices.
    int *d_A, *d_B; // Memoria en GPU
    cudaMalloc(&d_A, totalElementos * sizeof(int));
    cudaMalloc(&d_B, totalElementos * sizeof(int));

    // * Sacamos las propiedades del dispositivo
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades, 0);
    int tamanio_bloque = propiedades.maxThreadsPerBlock;
    
    // Mandar a proporciones del numero de hilos totales que tiene el dispositivo
    int num_bloques = (totalElementos + tamanio_bloque - 1) / tamanio_bloque; // esto nos sirve para dividir el trabajo en bloques de hilos

    cudaMemcpy(d_A, h_A, totalElementos * sizeof(int), cudaMemcpyHostToDevice); // * Copiar de CPU a GPU

    clock_t start = clock();
    transponer<<<num_bloques, tamanio_bloque>>>(d_A, d_B, m, n);
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    double s = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Tiempo de ejecución: %f s\n", s);

    cudaMemcpy(h_B, d_B, totalElementos * sizeof(int), cudaMemcpyDeviceToHost); // * Copiar de GPU a CPU

    if (m <= 20 && n <= 20)
    {
        printf("Matriz original:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", h_A[i * n + j]);
            }
            printf("\n");
        }

        printf("Matriz transpuesta:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%d ", h_B[i * m + j]);
            }
            printf("\n");
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}

// Normal
// * Resaltado
// ! Peligro
// ? Cómo
// TODO: 