// En una linea de ensamblaje de productos, existen tres tipos de trabajadores: cortadores, soldadores y pintores. Los cortadores cortan las piezas, los soldadores las sueldan y los pintores las pintan. Existen 3 tipos de trabajadores: cortadores, soldadores y pintores. Los cortadores cortan las piezas, los soldadores las sueldan y los pintores las pintan. Existen 3 tipos de trabajadores: cortadores, soldadores y pintores. Los cortadores cortan las piezas, los soldadores las sueldan y los pintores las pintan. Existen 3 tipos de trabajadores: cortadores, soldadores y pintores. Los cortadores cortan las piezas, los soldadores las sueldan y los pintores las pintan. Existen 3 tipos de trabajadores: cortadores, soldadores y pintores. Los cortadores cortan las piezas, los soldadores las sueldan y los pintores las pintan. Existen 3 tipos de trabajadores: cortadores, soldadores y pintores. Los cortadores cortan las piezas, los soldadores las sueldan y los pintores las pintan. Existen 3 tipos de trabajadores: cortadores, soldadores y pintores. Los cortadores cortan las piezas, los soldadores las sueldan y los pintores las pintan. Existen 3 tipos de trabajadores: cortadores, soldadores y pintores. 
// Cortadores: Se encargan de cortar la spiezas necesarias para el sensamblaje.
// Soldadores: Se encargan de soldar las piezas cortadas.
// Pintores: Se encargan de pintar las piezas soldadas.

// Para asegurar el flujo correcto de trabajo, se deben cumplir las siguinetes restricciones:
// los cortadores no pueden comenzxar con un producto nuevo hasta que los pintores hayan terminado de pintar el producto anterior.
// Los soldadores no pueden comenzar a soldar un producto hasta que los cortadores hayan terminado de cortar todas las piezas necesarias.
// Los pintores no pueden comenzar a pintar un producto hasta que los soldadores hayan terminado de soldar todas las piezas necesarias.

// Se deben ensamblar un total de M productos en paralelo, con multiples trabajadores en cada etapa.

// Implementar un programa en C usando pthreads, semaforos, barreras y variables de condicion para coordinar el trabajo de los cortadores, soldados y pintores. El programa debe:
// Crear hilos para representar a los trabajadores.
// Sincronizar su ejecucion para respetar el flujo de trabajo
// Repetir el proceso de ensamblaje M veces.

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define M 5  // Número de productos a ensamblar

sem_t cortadores_listos;
sem_t soldadores_listos;
sem_t pintores_listos;

int productos_terminados = 0;


void *cortar(void *arg) {
    int * id = (int *)arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&pintores_listos);  // Espera a que los pintores terminen
        printf("Cortador %d cortando piezas para el producto %d\n", *id, i + 1);
        sleep(1);
        sem_post(&cortadores_listos); // Sem_post suma 1 al semaforo cortadores_listos para que los soldadores puedan empezar
    }
    free(arg);
    return NULL;
}

void *soldar(void *arg) {
    int * id = (int *)arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&cortadores_listos);  // Espera a que los cortadores terminen
        printf("Soldador %d soldando piezas del producto %d\n", *id, i + 1);
        sleep(1);
        sem_post(&soldadores_listos); // Libera a los pintores
    }
    free(arg);
    return NULL;
}

void *pintar(void *arg) {
    int * id = (int *)arg;
    for (int i = 0; i < M; i++) {
        sem_wait(&soldadores_listos);  // Espera a que los soldadores terminen
        printf("Pintor %d pintando el producto %d\n", *id, i + 1);
        sleep(1);
        
        sem_post(&pintores_listos); // Permite a los cortadores empezar un nuevo producto
    } 
    free(arg);
    return NULL;
}

int main(char argc, char *argv[]) {
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
    sem_init(&pintores_listos, 0, numero_pintores); // Inicialmente los pintores pueden empezar
    for (int i = 0; i < numero_cortadores; i++){
        int * id = malloc(sizeof(int));
        *id = i;
        pthread_create(&cortadores[i], NULL, cortar, (void *)id);
    }

    for (int i = 0; i < numero_soldadores; i++){
        int * id = malloc(sizeof(int));
        *id = i;
        pthread_create(&soldadores[i], NULL, soldar, (void *)id);
    }

    for (int i = 0; i < numero_pintores; i++){
        int * id = malloc(sizeof(int));
        *id = i;
        pthread_create(&pintores[i], NULL, pintar, (void *)id);
    }

    for (int i = 0; i < numero_cortadores; i++){
        pthread_join(cortadores[i], NULL);
    }

    for (int i = 0; i < numero_soldadores; i++){
        pthread_join(soldadores[i], NULL);
    }

    for (int i = 0; i < numero_pintores; i++){
        pthread_join(pintores[i], NULL);
    }

    sem_destroy(&cortadores_listos);
    sem_destroy(&soldadores_listos);
    sem_destroy(&pintores_listos);

    return 0;
}
