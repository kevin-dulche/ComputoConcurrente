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

void ejecutaInstruccion(std::string instruccion){
    std::cout << "PC: " << pc << "--> Ejecutando: " << instruccion << "\n";
    pc++;
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // esta funcion nos ayuda a dormir el hilo por un tiempo determinado
    // dormimos directamente el hilo de ejecucion
}

int main(int argc, char const *argv[]){

    std::vector<std::string> instrucciones = { // el vector es un arreglo dinamico, en este caso de strings
        "Cargar datos del proceso A",
        "Cargar datos al proceso B",
        "Sumar los datos del proceso A",
        "Multiplicar los datos del proceso B",
        "Sumar los resultados del proceso A con los del proceso B"
    };

    // vamos a recorrer cada instruccion y vamos a mandar a aumentar nuestro contador de programa
    for (auto &instruccion : instrucciones){ // auto es un "tipo de dato" en C++ que se encarga de inferir el tipo de dato de la variable
        // Vamos a llamar a ala funcion que aumente el contador de programa pero utilizando hilos
        ejecutaInstruccion(instruccion);
    }

    std::cout << "\nPC al terminar la ejecucion --> " << pc << "\n";

    return 0;
}