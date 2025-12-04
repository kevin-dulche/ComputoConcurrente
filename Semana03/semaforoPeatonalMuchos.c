#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define ROJO 0
#define VERDE 1

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int luz_semaforo = ROJO;

void * semaforo(void * arg)
{
    sleep(3);
    pthread_mutex_lock(&mutex);
    luz_semaforo = VERDE;
    printf("Semaforo: Luz verde\n");
    sleep(2);
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void * peaton(void * arg)
{
    pthread_mutex_lock(&mutex);
    printf("Peaton %d: Esperando a que cambie la luz\n", *((int *)arg));
    while (luz_semaforo == ROJO)
        pthread_cond_wait(&cond, &mutex);
        printf("Peaton %d: Puedo cruzar\n", *((int *)arg));
    pthread_mutex_unlock(&mutex);
    free(arg);
    pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        printf("Uso: %s <numero de peatones>\n", argv[0]);
        return -1;
    }
    pthread_t hilo_semaforo;
    pthread_t hilos_peatones[atoi(argv[1])];
    
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        int * id = (int *)malloc(sizeof(int));
        *id = i;
        pthread_create(&hilos_peatones[i], NULL, &peaton, (void *)id);
    }
    pthread_create(&hilo_semaforo, NULL, &semaforo, NULL);
    


    pthread_join(hilo_semaforo, NULL);
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        pthread_join(hilos_peatones[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    return 0;
}
