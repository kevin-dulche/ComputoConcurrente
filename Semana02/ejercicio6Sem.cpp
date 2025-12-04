#include <iostream>
#include <thread>
#include <cstdlib>
#include <semaphore.h>

// Creamos un semaforo
sem_t semaforo;

int balance = 0;

void suma(int iteraciones){
    for(int i = 0; i < iteraciones; ++i){
        sem_wait(&semaforo);
        balance++;
        sem_post(&semaforo);
        std::cout << "Balance en el hilo suma = " << balance << std::endl;
    }
}

void resta(int iteraciones){
    for(int i = 0; i < iteraciones; ++i){
        sem_wait(&semaforo);
        balance--;
        sem_post(&semaforo);
        std::cout << "Balance en el hilo resta = " << balance << std::endl;
    }
}

int main(int argc, char const *argv[]){
    if (argc != 2){
        std::cout << "Uso: " << argv[0] << " <número de iteraciones>" << std::endl;
        return -1;
    }

    int iteraciones = std::atoi(argv[1]);

    sem_init(&semaforo, 0, 1);
    std::thread sumador(suma, iteraciones);
    std::thread restador(resta, iteraciones);

    sumador.join();
    restador.join();

    sem_destroy(&semaforo);
    std::cout << "El balance es: " << balance << std::endl;
    return 0;
}