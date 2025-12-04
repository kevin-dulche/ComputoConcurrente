#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>  // biblioteca para trabajar con los parametros del main y usar la funcion atoi
#include <vector>

// #define TAMANIO 10


//int arreglo[TAMANIO]; Ahora la idea es que sea un vector

void calcularCuadrados(int inicio, int fin, int id, std::vector<int> &arreglo){
    for(int i=inicio;i<fin;++i){
        arreglo[i] *= arreglo[i];
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "\nYo soy el hilo " << id << " y el valor que modifique es: " << arreglo[i] << std::endl;
    }
}

int main(int argc, char const *argv[]){
    
    if(argc != 2){
        std::cout << "Uso: " << argv[0] << " TAMANIO" << std::endl;
        return 1;
    }

    std::vector<int> arreglo;
    
    int TAMANIO = atoi(argv[1]);
    
    arreglo.resize(TAMANIO); // el arreglo es global y lo redimensionamos para poderlo trabajar
    
    for(int i=0;i<TAMANIO;++i)
    arreglo[i] = i;
    
    std::cout << "El arreglo original es: " << std::endl;
    for (int i = 0; i < TAMANIO; i++){
        std::cout << arreglo[i] << " ";
    }

    int mitad = TAMANIO / 2;

    std::thread hilo1(calcularCuadrados, 0, mitad, 1, std::ref(arreglo));
    std::thread hilo2(calcularCuadrados, mitad, TAMANIO, 2, std::ref(arreglo));

    hilo1.join();
    hilo2.join();

    std::cout << "\nEl arreglo modificado es: " << std::endl;
    for (int i = 0; i < TAMANIO; i++)
        std::cout << arreglo[i] << " ";
    std::cout << std::endl;

    return 0;
}

// std::thread t1([=](){
//         for(int i=0;i<TAMANIO;++i)
//             arreglo[i] *= 2;
//     });