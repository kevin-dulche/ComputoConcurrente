#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std;

// Kernel CUDA para la multiplicación de matrices
__global__ void multiplicarMatrices(int *A, int *B, int *C, int N) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < N && columna < N) {
        int suma = 0;
        for (int k = 0; k < N; k++) {
            suma += A[fila * N + k] * B[k * N + columna];
        }
        C[fila * N + columna] = suma;
    }
}

// Multiplicación secuencial en CPU
void multiplicarCPU(int *A, int *B, int *C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int suma = 0;
            for (int k = 0; k < N; ++k)
                suma += A[i * N + k] * B[k * N + j];
            C[i * N + j] = suma;
        }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamano N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int size = N * N * sizeof(int);

    // Asignación en host
    int *h_A = (int*) malloc(size);
    int *h_B = (int*) malloc(size);
    int *h_C_CPU = (int*) malloc(size);
    int *h_C_GPU = (int*) malloc(size);

    // Inicialización
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    // CPU (secuencial)
    auto start_cpu = chrono::high_resolution_clock::now();
    multiplicarCPU(h_A, h_B, h_C_CPU, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> dur_cpu = end_cpu - start_cpu;

    // Asignación en device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Dimensiones de ejecución
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                    (N + blockSize.y - 1) / blockSize.y);

    // GPU (paralelo)
    auto start_gpu = chrono::high_resolution_clock::now();
    multiplicarMatrices<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> dur_gpu = end_gpu - start_gpu;

    // Copiar resultado a host
    cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost);

    // Validación (opcional)
    bool correcto = true;
    for (int i = 0; i < N * N; ++i) {
        if (h_C_CPU[i] != h_C_GPU[i]) {
            correcto = false;
            break;
        }
    }

    printf("Validación: %s\n", correcto ? "Correcta" : "Incorrecta");
    printf("Tiempo CPU: %.2f ms\n", dur_cpu.count());
    printf("Tiempo GPU: %.2f ms\n", dur_gpu.count());

    // Liberación
    free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}