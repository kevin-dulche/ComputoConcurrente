#include <iostream>
#include <thread> // no pthread.h si no solamente thread
// esta biblioteca pertenece al estandar de C++ y permite trabajar con hilos

#include <vector> // es un EDL (Estructura de Datos Lineal) que nos ayuda a trabajar como un arreglo tradicional pero son dinamicos 
// (en tamaño) y ademas se guardan en el heap.

#include <string> // para trabajar con strings

#include <chrono> // esta biblioteca nos ayuda a trabajar con el tiempo, pero de manera nativa en C++. 
// Es menjor que la biblioteca time.h de C

// pthread es el estandar POSIX y trabaja mucho mejor en sistemas basados en UNIX.
// Sin embargo, thread como pertenece al estandar al escalar de C++, es mas escalable y portable.

int pc = 0; // variable global que simula el contador de programa (program counter, PC)

void ejecutaInstruccionA(){
    std::cout << "Estoy en la funcion A y el PC es : " << pc << std::endl;
    pc++;
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // esta funcion nos ayuda a dormir el hilo por un tiempo determinado
    // dormimos directamente el hilo de ejecucion
}

void ejecutaInstruccionB(){
    std::cout << "Estoy en la funcion B y el PC es : " << pc << std::endl;
    pc++;
    std::this_thread::sleep_for(std::chrono::milliseconds(800)); // esta funcion nos ayuda a dormir el hilo por un tiempo determinado
    // dormimos directamente el hilo de ejecucion
}

int main(int argc, char const *argv[]){

    std::cout << "\nIniciando la ejecucion del programa :)\n";
    std::cout << "\nPC --> " << pc << "\n";
    pc++;
    ejecutaInstruccionA();
    std::cout << "PC --> " << pc << "\n";
    std::cout << "Regresando el PC de la funcion A --> " << pc << "\n";
    pc++;
    ejecutaInstruccionB();
    std::cout << "Regresando el PC de la funcion B --> " << pc << "\n";
    pc++;
    std::cout << "\nEl PC al final es --> " << pc << "\n";
    return 0;
}