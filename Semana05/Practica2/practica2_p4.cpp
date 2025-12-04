/*
    Problema 4: Completa el programa
    Completa el siguiente código en C++ con std::thread para que realice correctamente la suma de matrices de forma concurrente
*/
#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib> 
#include <ctime> 
#include <mutex>  

using namespace std;
mutex cout_mutex;

void sumaMatrices(float *a, float *b, float *c, int filas, int columnas, int inicio, int fin, int id) {
    {
        lock_guard<mutex> lock(cout_mutex);
        cout<<"\nSoy el hilo "<<id<<" y voy a comenzar a sumar mis elementos\n";
    }
    for (int i = inicio; i < fin; i++) {
        for (int j = 0; j < columnas; j++) {
            int indice = i * columnas + j;
            c[indice] = a[indice] + b[indice];
        }
    }
    {
        lock_guard<mutex> lock(cout_mutex);
        cout <<"\nSoy el hilo "<<id<<" y termine de sumar mis elementos\n";
    }
}

int main() {
    srand(time(NULL));
    int filas = 4, columnas = 4;
    int total = filas * columnas;

    //se reserva memoria dinámica para las matrices con "new"
    float *a = new float[total];
    float *b = new float[total];
    float *c = new float[total];

    for (int i = 0; i < total; i++) {
        a[i] = static_cast<float>(rand() % 10);
        b[i] = static_cast<float>(rand() % 10);
    }

    cout<<"Matriz A:\n";
    for (int i = 0; i < total; i++) {
        cout << a[i] << " ";
        if ((i + 1) % columnas == 0) cout<<endl;
    }

    cout<<"\nMatriz B:\n";
    for (int i = 0; i < total; i++) {
        cout<<b[i]<<" ";
        if ((i + 1) % columnas == 0) cout<<endl;
    }

    int num_hilos = 2;
    vector<thread> hilos;
    int filas_por_hilo = filas / num_hilos;

    for (int i = 0; i < num_hilos; i++) {
        int inicio = i * filas_por_hilo;
        int fin = (i == num_hilos - 1) ? filas : (i + 1) * filas_por_hilo;
        hilos.emplace_back(sumaMatrices, a, b, c, filas, columnas, inicio, fin, i + 1);
    }

    for (auto &hilo : hilos) hilo.join();

    cout<<"\nMatriz resultante C:\n";
    for (int i = 0; i < total; i++) {
        cout<<c[i]<<" ";
        if ((i + 1) % columnas == 0) cout<<endl;
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
/*
    Aclaraciones:
    Para evitar la condición de carrera al usar std::cout en los hilos se agregó un mutex
    Hubo un cambio en el código, ya que al hacer uso de memoria dinámica (como con malloc en C) se hizo uso de new
*/