#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

typedef struct{
    int inicio;
    int incremento;
    int fin;
    int id;
} parametros_hilos;

void * imprimirIncremento(void *arg){
    parametros_hilos *philo = (parametros_hilos *) arg;
    for(int i = philo->inicio; i <= philo->fin; i += philo->incremento){
        printf("\nYo soy el hilo %d y el contador es: %d\n", philo->id, i);
        sleep(1);
    }
    return NULL;
}

int main(int argc, char const *argv[]){
    pthread_t hilo1, hilo2;
    parametros_hilos philo1;

    philo1.inicio = 1;
    philo1.incremento = 1;
    philo1.fin = 10;
    philo1.id = 1;

    parametros_hilos philo2 = {2, 2, 10, 2};

    pthread_create(&hilo1, NULL, imprimirIncremento, (void *) &philo1);
    pthread_create(&hilo2, NULL, imprimirIncremento, (void *) &philo2);

    pthread_join(hilo1, NULL);
    pthread_join(hilo2, NULL);

    printf("Fin del programa\n");

    return 0;
}
