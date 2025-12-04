#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

int pc_main = 0; // contador de programa del main

int pc_suma = 0; // contador de programa de la funcion suma

void suma(std::vector<int> numeros){
    std::cout << "\nIniciando el proceso de la suma con un PC (suma) --> " << pc_suma << "\n";
    int s = 0;
    for(int i = 0; i < numeros.size(); i++){
        s += numeros[i];
        pc_suma++;
        std::cout << "\nPC --> " << pc_suma << " sumando el numero " << numeros[i] << " y el total de la suma es: " << s << "\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

int main(int argc, char const *argv[])
{
    std::vector<int> numeros = {1, 2, 3, 4, 5};
    std::cout << "\nIniciando la ejecucion del programa\n";
    pc_main++;
    suma(numeros);
    std::cout << "\nPC --> " << pc_main << "\n";
    return 0;
}
