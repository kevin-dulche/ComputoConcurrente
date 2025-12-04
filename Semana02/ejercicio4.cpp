#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>

#define inicio 1
#define meta 30

int ganador=-1;

void brinca(int * id){
    int posicion = inicio;
    while(posicion < meta && ganador == -1){
        posicion += rand() % 10 + 1;//valores entre 1 y 10
        std::cout << "\nSoy la ranita " << *id << " y estoy en la posicion " << posicion;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    
    if(posicion >= meta && ganador == -1){
        ganador = *id;
    }
}

int main(int argc, char const *argv[]){

    int ranas = atoi(argv[1]);
    srand(time(NULL));
    std::cout << "\nVoy a mandar a competir a " << ranas << " ranas\n";

    std::thread ranitas[ranas];

    for(int i=0;i<ranas;++i){
        int * id = (int *) malloc(sizeof(int));
        * id = i;
        ranitas[i] = std::thread(brinca, id);
    }

    for(int i=0;i<ranas;++i)
        ranitas[i].join();

    std::cout << "\nLa ranita ganadora es: " << ganador;

    return 0;
}