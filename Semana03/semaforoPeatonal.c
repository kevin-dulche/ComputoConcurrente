#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define ROJO 0
#define VERDE 1

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int luz_semaforo = ROJO;

void * policia(void * arg)
{
    sleep(3);
    pthread_mutex_lock(&mutex);
    luz_semaforo = VERDE;
    printf("Policia: Luz verde\n");
    sleep(2);
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void * peaton(void * arg)
{
    pthread_mutex_lock(&mutex);
    printf("Peaton: Esperando a que cambie la luz\n");
    while (luz_semaforo == ROJO)
        pthread_cond_wait(&cond, &mutex);
    pthread_mutex_unlock(&mutex);
    printf("Peaton: Puedo cruzar\n");
    pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
    pthread_t hilo_policia, hilo_peaton;
    pthread_create(&hilo_peaton, NULL, &peaton, NULL);
    pthread_create(&hilo_policia, NULL, &policia, NULL);

    pthread_join(hilo_peaton, NULL);
    pthread_join(hilo_policia, NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    return 0;
}
