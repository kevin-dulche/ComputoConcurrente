// Ejercicio de repaso
// Debes hacer que el programa haga que crucen todos los peatones pero no al mismo tiempo... 
// es decir, no puedes usar broadcast, sino signal de uno en uno.
// * Kevin Uriel Dulche Jaime

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define ROJO 0
#define VERDE 1

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int luz_semaforo = ROJO; // 0 = rojo, 1 = verde
int peatones_en_espera = 0; // Numero de peatones esperando
int total_peatones; // Numero total de peatones

/**
 * @brief Funcion que simula el semaforo
 * @param arg: argumento de tipo void * que no se usa
 * @return void *: NULL
 */
void * semaforo(void * arg)
{
    sleep(3); // Esperar 3 segundos
    pthread_mutex_lock(&mutex); // Bloquear el mutex
    while (peatones_en_espera < total_peatones){ // Mientras haya peatones que no han pedido paso, {
                                                //esto para esperar a que todos los peatones pidan paso
        pthread_mutex_unlock(&mutex); // Desbloquear el mutex
        sleep(0.2);
        pthread_mutex_lock(&mutex); // Bloquear el mutex
    }
    luz_semaforo = VERDE; // Cambiar la luz del semaforo a verde cuando no haya peatones sin pedir paso
    printf("Semaforo: Luz verde\n");
    sleep(2);
    
    while (peatones_en_espera > 0)  // Mientras haya peatones esperando
    {
        pthread_cond_signal(&cond); // Dar paso a un peaton
        peatones_en_espera--; // Disminuir el numero de peatones esperando
        pthread_mutex_unlock(&mutex); // Desbloquear el mutex
        //printf("Semaforo: Peaton cruzando\n");
        //sleep(2);
        pthread_mutex_lock(&mutex); // Bloquear el mutex
    }

    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

/**
 * @brief Funcion que simula un peaton
 * @param arg: argumento de tipo void * que indica el numero de peaton
 * @return void *: NULL
 */
void * peaton(void * arg)
{
    pthread_mutex_lock(&mutex); // Bloquear el mutex
    printf("Peaton %d: Esperando a que cambie la luz\n", *((int *)arg));
    peatones_en_espera++; // Aumentar el numero de peatones esperando
    
    while (luz_semaforo == ROJO) // Mientras la luz del semaforo sea roja
        pthread_cond_wait(&cond, &mutex); // Esperar a que la luz cambie a verde

    printf("Peaton %d: Cruzando\n", *((int *)arg));
    sleep(2);
    printf("Peaton %d: He cruzado\n", *((int *)arg));
    pthread_mutex_unlock(&mutex); // Desbloquear el mutex
    free(arg); // Liberar memoria
    pthread_exit(NULL);
}

/**
 * @brief Funcion principal
 * @param argc: numero de argumentos
 * @param argv: argumentos de la funcion
 * @return int: 0 si termina correctamente, -1 si no se ingresan los argumentos correctos
 */
int main(int argc, char const *argv[])
{
    if (argc != 2) // Si no se ingresan los argumentos correctos
    {
        printf("Uso: %s <numero de peatones>\n", argv[0]);
        return -1;
    }

    pthread_t hilo_semaforo;
    pthread_t hilos_peatones[atoi(argv[1])];
    total_peatones = atoi(argv[1]);
    
    // Crear hilos
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        int * id = (int *)malloc(sizeof(int));
        *id = i;
        pthread_create(&hilos_peatones[i], NULL, &peaton, (void *)id);
    }
    pthread_create(&hilo_semaforo, NULL, &semaforo, NULL);


    // Esperar a que terminen los hilos
    pthread_join(hilo_semaforo, NULL);
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        pthread_join(hilos_peatones[i], NULL);
    }

    // Destruir mutex y cond
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    return 0;
}