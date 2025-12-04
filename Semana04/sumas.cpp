// ! Revisar el codigo, no sirve

#include <iostream>
#include <thread>
#include <chrono>
#include <condition_variable> // esta biblioteca es necesaria para el uso de variables de condición
#include <mutex> // esta biblioteca es necesaria para el uso de mutex

std::mutex mtx;
std::condition_variable cv;

int suma_pares = 0;
int suma_impares = 0;

int numero;

bool par=0, impar=0, terminado_pares=0, terminado_impares=0;

void calculoPares()
{
    for (int i = 0; i <= 100; i += 2)
    {
        std::lock_guard<std::mutex> lock(mtx); // esto nos sirve para poder manejar de mejor forma la liberacion "semi-automatica" del mutex
        numero = i;
        suma_pares += numero;
        par = 1;
        cv.notify_one(); // esto nos sirve para notificar a la variable de condición que se ha terminado de ejecutar una tarea
    }
    std::lock_guard<std::mutex> lock(mtx);
    terminado_pares = 1;
    cv.notify_one();
}

void calculoImpares()
{
    for (int i = 1; i <= 100; i += 2)
    {
        std::lock_guard<std::mutex> lock(mtx); // esto nos sirve para poder manejar de mejor forma la liberacion "semi-automatica" del mutex
        numero = i;
        suma_impares += numero;
        impar = 1;  
    }
    std::lock_guard<std::mutex> lock(mtx);
    terminado_impares = 1;
    cv.notify_one();
}

void imprimir()
{
    while(true){
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] {return par||impar||terminado_pares&&terminado_impares;}); 
        //el wait hace que el mutex o los mutex asociados se liberen de forma temporal
        if (par)
        std::cout << "Numero agregado a la suma de pares:" << numero << " y la suma total es: " << suma_pares << std::endl;
        par = 0;
        if (impar)
        std::cout << "Numero agregado a la suma de impares:" << numero << " y la suma total es: " << suma_impares << std::endl;
        impar = 0;
        if (terminado_pares && terminado_impares)
        {
            std::cout << "La suma total de los pares es: " << suma_pares << std::endl;
            std::cout << "La suma total de los impares es: " << suma_impares << std::endl;
            break;
        }
    }
}

int main(int argc, char const *argv[])
{
    std::thread hilo_pares(calculoPares);
    std::thread hilo_impares(calculoImpares);
    std::thread hilo(imprimir);

    return 0;
}