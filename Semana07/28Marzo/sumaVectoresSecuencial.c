#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Asigna memoria para los arreglos A, B y C.
 * @param A Puntero al arreglo A.
 * @param B Puntero al arreglo B.
 * @param C Puntero al arreglo C.
 * @param n Tamaño de los arreglos.
 * @return void
 * @note Asigna memoria dinámica para los arreglos A, B y C utilizando malloc.
 *       Si la asignación de memoria falla, se imprime un mensaje de error y se termina el programa.
 */
void asignarMemoria(float **A, float **B, float **C, int n)
{
    printf("Asignando memoria para el arreglo A\n");
    *A = (float *)malloc(n * sizeof(float));
    if (*A == NULL)
    {
        printf("Error al asignar memoria para A\n");
        exit(1);
    }
    printf("Asignando memoria para el arreglo B\n");
    *B = (float *)malloc(n * sizeof(float));
    if (*B == NULL)
    {
        printf("Error al asignar memoria para B\n");
        free(*A);
        exit(1);
    }
    printf("Asignando memoria para el arreglo C\n");
    *C = (float *)malloc(n * sizeof(float));
    if (*C == NULL)
    {
        printf("Error al asignar memoria para C\n");
        free(*A);
        free(*B);
        exit(1);
    }
}

/**
 * @brief Inicializa los arreglos A y B con números aleatorios.
 * @param A Arreglo A.
 * @param B Arreglo B.
 * @param n Tamaño de los arreglos.
 * @return void
 * @note Inicializa los arreglos A y B con números aleatorios entre 0 y 1.
 *       Utiliza la función rand() para generar los números aleatorios.
 *       La semilla para la generación de números aleatorios se establece utilizando time(NULL).
 *       Esto asegura que los números generados sean diferentes en cada ejecución del programa.
 */
void inicializarArreglos(float *A, float *B, int n)
{
    srand(time(NULL));
    printf("Inicializando los arreglos A y B con números aleatorios\n");
    for (int i = 0; i < n; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
}

void liberarMemoria(float *A, float *B, float *C)
{
    free(A);
    free(B);
    free(C);
}

/**
 * @brief Suma los elementos de los arreglos A y B y almacena el resultado en C.
 * @param A Arreglo A.
 * @param B Arreglo B.
 * @param C Arreglo C.
 * @param n Tamaño de los arreglos.
 * @return void
 * @note Suma los elementos de los arreglos A y B y almacena el resultado en C.
 *       Utiliza un bucle for para recorrer los elementos de los arreglos.
 */
void sumavectores(float *A, float *B, float *C, int n)
{
    printf("Sumando los elementos de los arreglos A y B\n");
    for (int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        printf("Uso: %s <tamaño del arreglo>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]); //tamaño del arreglo
    
    float *A, *B, *C;

    asignarMemoria(&A, &B, &C, n);
    inicializarArreglos(A, B, n);

    clock_t inicio, fin;
    inicio = clock();
    sumavectores(A, B, C, n);
    fin = clock();
    double tiempo = (double)(fin - inicio) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecución es de: %f segundos\n", tiempo);

    liberarMemoria(A, B, C);

    return 0;
}