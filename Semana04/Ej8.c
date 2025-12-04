#include <stdio.h>
#include <pthread.h>

#define BUFFER_SIZE 10

int buffer[BUFFER_SIZE];

int count = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t nolleno = PTHREAD_COND_INITIALIZER;

void * productor(){
    for (int i = 0; i < N; ++i)
    {
        while(count == BUFFER_SIZE)
            // Esperar
            pthread_cond_wait(&nolleno, &mutex);
        buffer[count] = i;
    }
}

void * consumidor(){
    for (int i = 0; i < N; ++i)
    {
        while(count == 0)
            // Esperar si el buffer esta vacio
            //int item = buffer[--count];
        pthread_cond_signal(&nolleno);
        
    }
}

int main(int argc, char const *argv[])
{
    pthread_t hilo_productor, hilo_consumidor;
    pthread_create(&hilo_productor, NULL, productor, NULL);
    pthread_create(&hilo_consumidor, NULL, consumidor, NULL);
    pthread_join(hilo_productor, NULL);
    pthread_join(hilo_consumidor, NULL);
    return 0;
}
