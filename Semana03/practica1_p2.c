#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <semaphore.h>

sem_t palillos[5];

void *cenar(void *arg)
{
    int id = *(int *)arg;
    int izquierda = id;
    int derecha = (id + 1) % 5;

    int primero, segundo;
    if (izquierda < derecha)
    {
        primero = izquierda;
        segundo = derecha;
    }
    else
    {
        primero = derecha;
        segundo = izquierda;
    }

    for (int i = 0; i < 5; i++)
    {
        printf("Filosofo %d esta pensando...\n", id);
        sleep(rand() % 3 + 1);

        // printf("Filosofo %d quiere comer\n", id);
        sem_wait(&palillos[primero]);
        printf("Filosofo %d tomo el palillo %d\n", id, primero);

        sem_wait(&palillos[segundo]);
        printf("Filosofo %d tomo el palillo %d y esta comiendo\n", id, segundo);

        sleep(rand() % 2 + 1);

        sem_post(&palillos[segundo]);
        sem_post(&palillos[primero]);
        printf("Filosofo %d termino de comer y solto los palillos\n", id);
    }
    return NULL;
}

int main()
{
    pthread_t filosofos[5];
    int ids[5];

    for (int i = 0; i < 5; i++)
    {
        sem_init(&palillos[i], 0, 1);
    }

    for (int i = 0; i < 5; i++)
    {
        ids[i] = i;
        pthread_create(&filosofos[i], NULL, cenar, &ids[i]);
    }

    for (int i = 0; i < 5; i++)
    {
        pthread_join(filosofos[i], NULL);
    }

    printf("\nLa cena termino!\n");

    for (int i = 0; i < 5; i++)
    {
        sem_destroy(&palillos[i]);
    }

    return 0;
}
