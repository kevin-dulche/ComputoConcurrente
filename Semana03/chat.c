#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define ROJO 0
#define VERDE 1

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int luz_semaforo = ROJO;
int peatones_esperando = 0; // Contador de peatones esperando

void *semaforo(void *arg)
{
    sleep(3);
    pthread_mutex_lock(&mutex);
    luz_semaforo = VERDE;
    printf("Semaforo: Luz verde\n");

    // Despierta a cada peatón uno por uno
    while (peatones_esperando > 0)
    {
        pthread_cond_signal(&cond);
        peatones_esperando--;
        sleep(1); // Pequeña pausa para simular que cruzan en orden
    }

    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void *peaton(void *arg)
{
    int id = *((int *)arg);
    free(arg); // Liberar memoria antes de bloquearse

    pthread_mutex_lock(&mutex);
    printf("Peaton %d: Esperando a que cambie la luz\n", id);
    peatones_esperando++; // Aumenta el contador de peatones esperando

    while (luz_semaforo == ROJO)
        pthread_cond_wait(&cond, &mutex); // Espera a la señal

    pthread_mutex_unlock(&mutex);

    printf("Peaton %d: Puedo cruzar\n", id);
    pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        printf("Uso: %s <numero de peatones>\n", argv[0]);
        return -1;
    }

    int num_peatones = atoi(argv[1]);
    if (num_peatones <= 0)
    {
        printf("Error: El numero de peatones debe ser mayor que 0\n");
        return -1;
    }

    pthread_t hilo_semaforo;
    pthread_t hilos_peatones[num_peatones];

    for (int i = 0; i < num_peatones; i++)
    {
        int *id = malloc(sizeof(int));
        if (!id)
        {
            perror("Error al asignar memoria");
            return -1;
        }
        *id = i;
        pthread_create(&hilos_peatones[i], NULL, &peaton, id);
    }

    pthread_create(&hilo_semaforo, NULL, &semaforo, NULL);

    pthread_join(hilo_semaforo, NULL);
    for (int i = 0; i < num_peatones; i++)
    {
        pthread_join(hilos_peatones[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    return 0;
}