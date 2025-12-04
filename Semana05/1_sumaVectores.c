#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

typedef struct{
    float * A;
    float * B;
    int inicio;
    int fin;
    int id;
} parametros;

void * suma_vectores(void * args){
    parametros * argumentosSuma = (parametros *)args;
    for(int i = argumentosSuma->inicio; i < argumentosSuma->fin; i++){
        argumentosSuma->A[i] += argumentosSuma->B[i];
    }
    printf("Hilo %d terminado, sume del %d al %d\n", argumentosSuma->id, argumentosSuma->inicio, argumentosSuma->fin);
    pthread_exit(NULL);
}

int main(int argc, char const *argv[]){

    if(argc != 3){
        printf("Uso: %s <N> <num_hilos>\n", argv[0]);
        exit(-1);
    }

    int N = atoi(argv[1]);
    int num_hilos = atoi(argv[2]);
    int tamanioBloque = N / num_hilos; // ? tenemos que cada bloque es del tamanio correspondiente a la proporcion que le toca a cada hilo
    // ! Hay que verificar ue los valores sean divididos de tal manera que no perdamos informacion

    float * A, *B; // * trabajar con memoria dinamica dado que queremos que sean bastantes grandes

    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));

    srand(time(NULL));

    for(int i = 0; i < N; i++){
        A[i] = (float)rand()/RAND_MAX; // * Generamos un numero aleatorio entre 0 y 1
        B[i] = (float)rand()/RAND_MAX;
    }

    // TODO: Vamos a guardar los resultados en el vector A
    
    // * Vamos a crear los hilos
    pthread_t hilos[num_hilos];

    parametros argumentos[num_hilos];

    // imprimir los valores de A y B
    // for(int i = 0; i < N; i++){
    //     printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
    // }


    for(int i = 0; i < num_hilos; i++){
        argumentos[i].A = A;
        argumentos[i].B = B;
        argumentos[i].inicio = i * tamanioBloque;
        argumentos[i].fin = (i + 1) * tamanioBloque;
        argumentos[i].id = i;
        if (i == num_hilos - 1){
            argumentos[i].fin = N;
        }
        pthread_create(&hilos[i], NULL, suma_vectores, (void *)&argumentos[i]);
    }

    for(int i = 0; i < num_hilos; i++){
        pthread_join(hilos[i], NULL);
    }

    // imprimir los valores de A y B
    // for(int i = 0; i < N; i++){
    //     printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
    // } 

    free(A);
    free(B);

    return 0;
}