// Practica 1: Computo Concurrente
// Alumnos: 
// * Kevin Uriel Dulche Jaime
// * Gustavo Angel Morales Vigi
// * Bryan Tonatiuh Ochoa De La Cruz
// Fecha: 24 de febrero de 2025

// Problema 1: Lectores y Escritores

#include <stdio.h> // Incluir la libreria stdio para usar printf
#include <stdlib.h> // Incluir la libreria stdlib para usar atoi
#include <pthread.h> // Incluir la libreria pthread para usar hilos
#include <unistd.h> // Incluir la libreria unistd para usar sleep
#include <string.h> // Incluir la libreria string para usar sprintf

char libro[30] = "Este libro trata de ..."; // Declarar el libro de forma global

pthread_mutex_t mutex; // Declarar el mutex

/**
 * @brief Funcion que simula la lectura de un libro
 * @param arg: argumento de tipo void * que indica el numero de lector
 * @author Kevin Dulche
 */
void *leer(void *arg){
    for (int i = 0; i < 5; i++) // Leer 5 veces
    {
        pthread_mutex_lock(&mutex);
        printf("Lector %d leyendo: %s\n", *(int *)arg, libro);
        pthread_mutex_unlock(&mutex);
        sleep(1);
    }

    free(arg); // Liberar memoria
    pthread_exit(NULL);
}

/**
 * @brief Funcion que simula la escritura de un libro
 * @param arg: argumento de tipo void * que indica el numero de lectores del libro
 * @author Kevin Dulche
 */
void *escribir(void *arg){

    for (int i = 0; i < *(int *)arg; i++) // Escribir el libro
    {
        pthread_mutex_lock(&mutex);
        printf("Escritor escribiendo: %d\n", i);
        sprintf(libro, "Este libro trata de ... %d", i);
        sleep(2);
        pthread_mutex_unlock(&mutex);
        sleep(1);
    }

    pthread_exit(NULL);
}

/**
 * @brief Funcion principal
 * @param argc: numero de argumentos
 * @param argv: argumentos de la funcion
 * @return int: 0 si termina correctamente, -1 si no se ingresan los argumentos correctos
 */
int main(int argc, char const *argv[]){

    if (argc != 2){
        printf("Uso: %s <num_lectores>\n", argv[0]);
        exit(-1);
    }

    int num_lectores = atoi(argv[1]);
    pthread_t lectores[num_lectores];
    pthread_t escritor;

    pthread_create(&escritor, NULL, escribir, (void *)&num_lectores);

    for (int i = 0; i < num_lectores; i++){
        int * num = malloc(sizeof(int)); // Reservar memoria
        *num = i; // Asignar el valor
        pthread_create(&lectores[i], NULL, leer, (void *)num); // Pasar el numero de lector
    }
    
    // Esperar a que terminen los hilos
    pthread_join(escritor, NULL);

    for (int i = 0; i < num_lectores; i++){
        pthread_join(lectores[i], NULL);
    }

    // Destruir el mutex
    pthread_mutex_destroy(&mutex);

    return 0;
}