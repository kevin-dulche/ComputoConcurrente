#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

float *A, *B, *C; 
int n, num_hilos; 

void *sumaMatricesHilos(void *arg) {
    int idHilo = *(int *)arg;
    int fila = idHilo;  
    for (int i = 0; i < n; i++) {
        C[fila * n + i] = A[fila * n + i] + B[fila * n + i];
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        return 1;
    }

    srand(time(NULL));
    n = atoi(argv[1]); // Tamaño de la matriz (NxN)
    num_hilos = atoi(argv[2]); // Número de hilos igual a 'n'

    A = (float *)malloc(n * n * sizeof(float)); 
    B = (float *)malloc(n * n * sizeof(float)); 
    C = (float *)malloc(n * n * sizeof(float)); 

    // Inicializar matrices con valores aleatorios
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    pthread_t hilos[n];
    int id_hilos[n];

    clock_t inicio, fin;

    inicio = clock(); 
    // Crear hilos
    for (int i = 0; i < n; i++) {
        id_hilos[i] = i;
        pthread_create(&hilos[i], NULL, sumaMatricesHilos, &id_hilos[i]);
    }

    // Esperar a que los hilos terminen
    for (int i = 0; i < n; i++) {
        pthread_join(hilos[i], NULL);
    }

    fin = clock();

    printf("\nEl tiempo de ejecucion de la suma es: %f segundos\n", (double)(fin - inicio) / CLOCKS_PER_SEC);

    if (n<= 6)
    {
        printf("\nEl vector a = \n");
        for (int i = 0; i < n * n; i++)
        {
            printf("%.2f ", A[i]);
            if ((i + 1) % n == 0)
                printf("\n");
        }
        printf("\nEl vector b = \n");
        for (int i = 0; i < n * n; i++)
        {
            printf("%.2f ", B[i]);
            if ((i + 1) % n == 0)
                printf("\n");
        }
        printf("\nEl vector c = \n");
        for (int i = 0; i < n * n; i++)
        {
            printf("%.2f ", C[i]);
            if ((i + 1) % n == 0)
                printf("\n");
        }
        
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
