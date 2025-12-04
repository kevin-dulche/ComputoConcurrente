#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdlib.h>

#define M 5  // Número de productos a ensamblar

#define nCortadores 3
#define nSoldadores 3
#define nPintores 2

sem_t cortadores_listos;
sem_t soldadores_listos;
sem_t pintores_listos;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *cortar(void *arg) {
    int * id = (int *) arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&pintores_listos);  // Espera a que los pintores terminen el producto anterior
        printf("Cortador %d cortando piezas para el producto %d\n", *id, i + 1);
        sleep(1);
        sem_post(&cortadores_listos); // Libera a los soldadores
    }
    return NULL;
}

void *soldar(void *arg) {
    int * id = (int *) arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&cortadores_listos);  // Espera a que los cortadores terminen
        printf("Soldador %d soldando piezas del producto %d\n", *id, i + 1);
        sleep(1);
        sem_post(&soldadores_listos); // Libera a los pintores
    }
    return NULL;
}

void *pintar(void *arg) {
    int * id = (int *) arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&soldadores_listos);  // Espera a que los soldadores terminen
        printf("Pintor %d pintando el producto %d\n", *id, i + 1);
        sleep(1);
        sem_post(&pintores_listos); // Permite a los cortadores empezar un nuevo producto
    }
    return NULL;
}

int main() {
    pthread_t cortadores[nCortadores], soldadores[nSoldadores], pintores[nPintores];
    sem_init(&cortadores_listos, 0, 0);
    sem_init(&soldadores_listos, 0, 0);
    sem_init(&pintores_listos, 0, 1); // INiciado con 1 para que los cortadores puedan cortar desde el principio

    for (int i = 0; i < nCortadores; i++){
        int * id = malloc(sizeof(int));
        *id = i;
        pthread_create(&pintores[i], NULL, cortar, (void *)id);
    }
    for (int i = 0; i < nSoldadores; i++){
        int * id = malloc(sizeof(int));
        *id = i;
        pthread_create(&soldadores[i], NULL, soldar, (void *)id);
    }
    for (int i = 0; i < nPintores; i++){
        int * id = malloc(sizeof(int));
        *id = i;
        pthread_create(&pintores[i], NULL, pintar, (void *)id);
    }
    for (int i = 0; i < nCortadores; i++)
        pthread_join(cortadores[nPintores], NULL);
    for (int i = 0; i < nSoldadores; i++)
        pthread_join(soldadores[i], NULL);
    for (int i = 0; i < nPintores; i++)
        pthread_join(pintores[i], NULL);

    sem_destroy(&cortadores_listos);
    sem_destroy(&soldadores_listos);
    sem_destroy(&pintores_listos);
    pthread_mutex_destroy(&mutex);

    return 0;
}
