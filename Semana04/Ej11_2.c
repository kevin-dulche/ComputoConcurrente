#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define M 5  // Número de productos a ensamblar

sem_t cortadores_listos;
sem_t soldadores_listos;
sem_t pintores_listos;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int productos_terminados = 0;

void *cortar(void *arg) {
    int *id = (int *)arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&pintores_listos);  
        printf("Cortador %d cortando piezas para el producto %d\n", *id, i + 1);
        sleep(1);
        sem_post(&cortadores_listos);
    }
    free(arg);
    return NULL;
}

void *soldar(void *arg) {
    int *id = (int *)arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&cortadores_listos);  
        printf("Soldador %d soldando piezas del producto %d\n", *id, i + 1);
        sleep(1);
        sem_post(&soldadores_listos);
    }
    free(arg);
    return NULL;
}

void *pintar(void *arg) {
    int *id = (int *)arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&soldadores_listos);
        printf("Pintor %d pintando el producto %d\n", *id, i + 1);
        sleep(1);
        
        pthread_mutex_lock(&mutex);
        productos_terminados++;
        printf("Producto %d ensamblado.\n", productos_terminados);
        if (productos_terminados == M) {
            printf("Todos los productos han sido ensamblados.\n");
        }
        pthread_mutex_unlock(&mutex);
        
        sem_post(&pintores_listos);
    } 
    free(arg);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <num_cortadores> <num_soldadores> <num_pintores>\n", argv[0]);
        return 1;
    }

    int numero_cortadores = atoi(argv[1]);
    int numero_soldadores = atoi(argv[2]);
    int numero_pintores = atoi(argv[3]);

    pthread_t cortadores[numero_cortadores], soldadores[numero_soldadores], pintores[numero_pintores];
    
    sem_init(&cortadores_listos, 0, 0);
    sem_init(&soldadores_listos, 0, 0);
    sem_init(&pintores_listos, 0, 1); // Solo un pintor al inicio

    for (int i = 0; i < numero_cortadores; i++) {
        int *id = malloc(sizeof(int));
        *id = i;
        pthread_create(&cortadores[i], NULL, cortar, (void *)id);
    }

    for (int i = 0; i < numero_soldadores; i++) {
        int *id = malloc(sizeof(int));
        *id = i;
        pthread_create(&soldadores[i], NULL, soldar, (void *)id);
    }

    for (int i = 0; i < numero_pintores; i++) {
        int *id = malloc(sizeof(int));
        *id = i;
        pthread_create(&pintores[i], NULL, pintar, (void *)id);
    }

    for (int i = 0; i < numero_cortadores; i++) {
        pthread_join(cortadores[i], NULL);
    }

    for (int i = 0; i < numero_soldadores; i++) {
        pthread_join(soldadores[i], NULL);
    }

    for (int i = 0; i < numero_pintores; i++) {
        pthread_join(pintores[i], NULL);
    }

    sem_destroy(&cortadores_listos);
    sem_destroy(&soldadores_listos);
    sem_destroy(&pintores_listos);
    pthread_mutex_destroy(&mutex);

    return 0;
}
