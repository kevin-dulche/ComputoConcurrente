#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int senal_simulada = 0;

void * hilo_secundario(void * arg)
{
    printf("Hilo secundario ejecutandose\n");
    sleep(5);
    pthread_mutex_lock(&mutex);
    senal_simulada = 1;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
    pthread_t hilo;

    pthread_create(&hilo, NULL, &hilo_secundario, NULL);
    printf("Mande a ejecutar el hilo secundario, y esperare la senal\n");
    pthread_mutex_lock(&mutex);
    
    while (senal_simulada == 0)
    {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
    printf("Señal recibida\n");
    pthread_join(hilo, NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}
