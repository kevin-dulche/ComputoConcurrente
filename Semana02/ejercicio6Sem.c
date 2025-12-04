#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <semaphore.h> // Esta biblioteca nos sirve para trabajar con semáforos

int balance = 0;

sem_t semaforo; // declaramos un semáforo

void *suma(void *arg){
    int * iteraciones = (int *)arg;
    for(int i = 0; i < *iteraciones; ++i){
        sem_wait(&semaforo); // bloqueamos el semáforo
        balance++;
        sem_post(&semaforo); // desbloqueamos el semáforo
        printf("Balance en el hilo suma = %d\n", balance);
    }
    return NULL;
}

void *resta(void *arg){
    int * iteraciones = (int *)arg;
    for(int i = 0; i < *iteraciones; ++i){
        sem_wait(&semaforo); // bloqueamos el semáforo
        // * Como parametro recibe la dirección de memoria del semáforo
        // * Si el semáforo es mayor a 0, decrementa el semáforo y continua con
        // * la ejecución del hilo, si el semáforo es 0, el hilo se bloquea
        // * hasta que el semáforo sea mayor a 0
        balance--;
        sem_post(&semaforo); // desbloqueamos el semáforo
        printf("Balance en el hilo resta: %d\n", balance);
    }
    return NULL;
}

int main(int argc, char const *argv[]){
    if (argc != 2){
        printf("Uso: %s <número de iteraciones>\n", argv[0]);
        return -1;
    }

    sem_init(&semaforo, 0, 1); // inicializamos el semáforo en el main, dado que fuera de el no seria lo mas adecuado
    // * Los parametros del sem_init son:
    // * 1. Dirección de memoria del semáforo
    // * 2. El segundo parametro es 0, que indica que el semáforo es compartido entre hilos o procesos, 
    //      * si fuera 1 sería compartido entre procesos
    // * 3. El tercer parametro es el valor inicial del semáforo, en este caso 1

    int iteraciones = atoi(argv[1]);
    pthread_t sumador, restador;
    pthread_create(&sumador, NULL, suma, (void *)&iteraciones);
    pthread_create(&restador, NULL, resta, (void *)&iteraciones);
    pthread_join(sumador, NULL);
    pthread_join(restador, NULL);

    sem_destroy(&semaforo); // destruimos el semáforo
    // * Esta función libera los recursos asociados al semáforo
    printf("El balance es: %d\n", balance);
    return 0;
}