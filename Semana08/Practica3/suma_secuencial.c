#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void sumaMatricesSecuencial(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        C[i] = A[i] + B[i]; 
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        return 1;
    }

    srand(time(NULL));
    int n = atoi(argv[1]);
    
    float *A, *B, *C;
    A = (float *)malloc(n * n * sizeof(float)); 
    B = (float *)malloc(n * n * sizeof(float)); 
    C = (float *)malloc(n * n * sizeof(float)); 


    for (int i = 0; i < n * n; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t inicio, fin;
    inicio = clock();
    sumaMatricesSecuencial(A, B, C, n);
    fin = clock();

    printf("\nEl tiempo de ejecucion es: %f segundos\n", (double)(fin - inicio) / CLOCKS_PER_SEC);

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

    free(A); // Liberar memoria
    free(B);
    free(C);

    return 0;
}
