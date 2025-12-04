#include <stdio.h>
#include <pthread.h>

int contador = 1;

// TODO: Agregar un mutex para que los hilos no se interrumpan entre sí

void *incremento(void *arg){
    for(int i = 0; i < 10000; i++){
        contador++;
    }
    return NULL;
}

int main(){
    pthread_t hilo1, hilo2; // Tenemos el tipo pthread_t para declarar los hilos
    
    pthread_create(&hilo1, NULL, incremento, NULL); // Creamos el hilo 1, pasa la dirección de memoria del hilo, atributos del hilo, función a ejecutar y argumentos de la función

    pthread_create(&hilo2, NULL, incremento, NULL); // Creamos el hilo 2, pasa la dirección de memoria del hilo, atributos del hilo, función a ejecutar y argumentos de la función
    
    pthread_join(hilo1, NULL); // Espera a que el hilo 1 termine
    pthread_join(hilo2, NULL); // Espera a que el hilo 2 termine

    printf("El valor del contador es: %d\n", contador);
    return 0;
}