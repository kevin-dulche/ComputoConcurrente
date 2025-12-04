// A = 1, B = 2, C = 3, D = 4, E = 5, F = 6, G = 7, H = 8, I = 9, J = 10, K = 11, L = 12, 
// M = 13, N = 14, O = 15, P = 16, Q = 17, R = 18, S = 19, T = 20, U = 21, V = 22, W = 23, X = 24, Y = 25, Z = 26

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void productoElemento(int *A, int *B, int *C, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        C[idx] = A[idx] * B[idx];
    }
}

int main(int argc, char const *argv[])
{
    /* • Elige un tamaño de vector de acuerdo con la primera letra de tu nombre 
    (posición en el alfabeto multiplicada por 500; por ejemplo, si empieza con ’D’ sería 4 × 500 = 2000 elementos).
    */
    int K = 11;
    int tamanio = K * 500; // 11 * 500 = 5500
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    /* • Elige el número de hilos por bloque según la última letra de tu apellido paterno 
    (posición en el alfabeto multiplicada por 4; si es mayor al máximo permitido, ajusta al máximo de 1024).
    */
    int E = 5;
    int hilos = E * 4; // 5 * 4 = 20

    // Llenar los vectores A y B con valores aleatorios
    h_A = (int *)malloc(tamanio * sizeof(int));
    h_B = (int *)malloc(tamanio * sizeof(int));
    h_C = (int *)malloc(tamanio * sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < tamanio; i++)
    {
        h_A[i] = rand() % 100 + 1; // 1 - 100 
        h_B[i] = rand() % 100 + 1; // 1 - 100
    }

    /* • Implementa en CUDA un programa que calcule el producto elemento a elemento de dos vectores 
    (vector A * vector B = vector C), utilizando memoria dinámica.*/
    cudaMalloc(&d_A, tamanio * sizeof(int));
    cudaMalloc(&d_B, tamanio * sizeof(int));
    cudaMalloc(&d_C, tamanio * sizeof(int));

    cudaMemcpy(d_A, h_A, tamanio * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, tamanio * sizeof(int), cudaMemcpyHostToDevice);

    int bloques = (tamanio + hilos - 1) / hilos;

    // • Mide y muestra el tiempo de ejecución del kernel.
    clock_t start = clock();
    productoElemento<<<bloques, hilos>>>(d_A, d_B, d_C, tamanio);
    cudaDeviceSynchronize();
    clock_t end = clock();

    double tiempo = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecucion: %f s\n", tiempo);
    
    cudaMemcpy(h_C, d_C, tamanio * sizeof(int), cudaMemcpyDeviceToHost);

    // • Muestra los primeros y últimos 5 resultados del vector para verificar el funcionamiento.
    printf("Primeros 5 resultados del vector C: \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    printf("Ultimos 5 resultados del vector C: \n");
    for (int i = tamanio - 5; i < tamanio; i++)
    {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    /* • Incluye en tu entrega el código fuente, una breve explicación de cómo obtuviste el tamaño
    y la configuración del número de hilos, y capturas de pantalla de la ejecución.*/

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}