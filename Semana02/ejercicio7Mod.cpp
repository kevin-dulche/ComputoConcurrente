#include <iostream>
#include <thread>
#include <cstdlib>
#include <semaphore>

// Creamos un semaforo
std::counting_semaphore<1> semaforo(1);
// g++ nombre.cpp -o nombre -std=c++20
// std::binary_semaphore semaforo(1);

int balance = 0;

void suma(int iteraciones){
    for(int i = 0; i < iteraciones; ++i){
        semaforo.acquire();
        balance++;
        semaforo.release();
        std::cout << "Balance en el hilo suma = " << balance << std::endl;
    }
}

void resta(int iteraciones){
    for(int i = 0; i < iteraciones; ++i){
        semaforo.acquire();
        balance--;
        semaforo.release();
        std::cout << "Balance en el hilo resta = " << balance << std::endl;
    }
}

int main(int argc, char const *argv[]){
    if (argc != 2){
        std::cout << "Uso: " << argv[0] << " <número de iteraciones>" << std::endl;
        return -1;
    }

    int iteraciones = std::atoi(argv[1]);

    std::thread sumador(suma, iteraciones);
    std::thread restador(resta, iteraciones);

    sumador.join();
    restador.join();

    std::cout << "El balance es: " << balance << std::endl;
    return 0;
}