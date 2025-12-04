// para compilar es necesario algunas veces usar:
// gcc nombre.c -o nombre -lpthread
#include <stdio.h>
#include <pthread.h>
#include <time.h> // nos puede servir para domrir o pausar el proceso principal
// pero para dormir los procesos(hilos) o manejarlos con fork, exec, getpid,
// usamos una biblioteca llamada especial que es unistd

#include <unistd.h>
// vamos a tener una variable global para poder 
// compartir la informacion entre los procesos

void * muestraContador(void * arg){ 
    /* Estas funciones que ejecutan los hilos,
    necesitan recibir los parametros como un apuntador del tipo void * 
    */
   int * contador = (int *) arg;
    while (*contador < 10){
        printf("\nEl contador es: %d\n", *contador);
        sleep(1); // dormimos el proceso por 2 segundos
       // pthread_exit(NULL); // esto es para terminar el hilo
    }
    return NULL;
}

int main(int argc, char const *argv[]){
    int contador = 0;
    pthread_t hilo1; 
    /* es el tipo de dato hilo POSIX,
    y necesitamos declarar o definir cada uno de ellos.
    Con esto, lo que hacemos es asignar espacios de memoria para la pila (variables del hilo),
    registros del CPU (instrucciones) y el contador de progrma
    */

    pthread_create(&hilo1, NULL, muestraContador, (void *) &contador);
    for (int i = 0; i < 10; i++){
        printf("\nEstoy incrementando el contador :)\n");
        contador+=1;
        sleep(1); // este proceso (hilo) se duerme por 1 segundo
    }
    pthread_join(hilo1, NULL); // esperamos a que el hilo termine

    return 0;
}