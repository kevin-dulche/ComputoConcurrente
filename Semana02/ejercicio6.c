#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

int balance = 0;

pthread_mutex_t mutex;

void *suma(void *arg){
    int * iteraciones = (int *)arg;
    for(int i = 0; i < *iteraciones; ++i){
        pthread_mutex_lock(&mutex);
        balance++;
        pthread_mutex_unlock(&mutex);
        printf("Balance en el hilo suma = %d\n", balance);
    }
    return NULL;
}

void *resta(void *arg){
    int * iteraciones = (int *)arg;
    for(int i = 0; i < *iteraciones; ++i){
        pthread_mutex_lock(&mutex);
        balance--;
        pthread_mutex_unlock(&mutex);
        printf("Balance en el hilo resta: %d\n", balance);
    }
    return NULL;
}

int main(int argc, char const *argv[]){
    if (argc != 2){
        printf("Uso: %s <número de iteraciones>\n", argv[0]);
        return -1;
    }
    int iteraciones = atoi(argv[1]);
    pthread_t sumador, restador;
    pthread_create(&sumador, NULL, suma, (void *)&iteraciones);
    pthread_create(&restador, NULL, resta, (void *)&iteraciones);
    pthread_join(sumador, NULL);
    pthread_join(restador, NULL);
    printf("El balance es: %d\n", balance);
    return 0;
}