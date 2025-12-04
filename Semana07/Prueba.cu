// suma de matrices en CUDA
# include <iostream>
# include <cuda_runtime.h>// aqui deben cambiar esto porque , deben usar parametros
# define N 512 // tamanio de la matriz

// kernel de CUDA para la suma de matrices
__global__ void sumaMatrices (int *A , int *B , int *C , int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * n + i;

    if( i < n && j < n ) {
        C[idx] = A [idx] + B[idx];
    }
}

int main () {
    int size = N * N * sizeof(int);
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    // reservamos memoria en CPU
    h_A = (int *) malloc(size);
    h_B = (int *) malloc(size);
    h_C = (int *) malloc(size);

    // inicializamos matrices con valores aleatorios
    for(int i = 0; i < N * N ; i ++) {
        h_A [ i ] = rand () % 10;
        h_B [ i ] = rand () % 10;
    }

    // reservamos memoria en GPU
    cudaMalloc((void**) &d_A,size);
    cudaMalloc((void**) &d_B,size);
    cudaMalloc((void**) &d_C,size);

    // copiamos datos de CPU a GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // configuramos dimensiones del grid y los bloques
    dim3 blockSize(16 , 16);
    dim3 gridSize(( N + blockSize.x - 1) / blockSize.x , (N + blockSize.y - 1) / blockSize.y);

    // lanzamos el kernel en la GPU
    sumaMatrices<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // copiamos el resultado de GPU a CPU
    cudaMemcpy (h_C, d_C, size, cudaMemcpyDeviceToHost) ;

    // imprimimos algunos valores de la matriz resultante
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "C[N - 1][N - 1] = " << h_C[N * N - 1] << std::endl;

    // liberamos memoria
    free (h_A);
    free(h_B);
    free(h_C);
    cudaFree (d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}