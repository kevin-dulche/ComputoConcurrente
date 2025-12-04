#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void sumaVectores(int * A_dispositivo, int * B_dispositivo, int * C_dispositivo, int N){//n es la dimension
	int idHilo = blockIdx.x * blockDim.x + threadIdx.x;

	if(idHilo < N){
		C_dispositivo[idHilo] = A_dispositivo[idHilo] + B_dispositivo[idHilo];//cada hilo recibe un identificador de la posicion que va a trabajar
		}
}

int main(int argc, char * argv[]){
	srand(time(NULL));
	int dimension=atoi(argv[1]);//el tamaño que tendran los arreglos
	cudaEvent_t inicio, final;
	cudaEventCreate(&inicio);
	cudaEventCreate(&final);
	float tiempo;
	int *A_dispositivo, *B_dispositivo, *C_dispositivo;
	int *A_host, *B_host, *C_host;
	//TENGO EL MANEJO DE LOS ARREGLOS EN EL HOST
	A_host=(int *)malloc(dimension*sizeof(int));
	B_host=(int *)malloc(dimension*sizeof(int));
	C_host=(int *)malloc(dimension*sizeof(int));
	
	for(int i=0;i<dimension;++i){
		A_host[i]= 10 + rand() % 90;
		B_host[i]= 10 + rand() % 90;
	}	

	//declaramos de la memoria en el dispositivo
	cudaMalloc(&A_dispositivo,dimension*sizeof(int));
	cudaMalloc(&B_dispositivo,dimension*sizeof(int));
	cudaMalloc(&C_dispositivo,dimension*sizeof(int));
	
	//movemos la memoria del host al dispositivo
	cudaMemcpy(A_dispositivo,A_host, dimension*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(B_dispositivo,B_host, dimension*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(C_dispositivo,C_host, dimension*sizeof(int),cudaMemcpyHostToDevice);
	
	//sacamos las propiedades de nuestro dispositivo
	cudaDeviceProp propiedades;
	cudaGetDeviceProperties(&propiedades,0);
    int hilos = atoi(argv[2]);
	int tamanio_bloque = hilos; //numero de hilos por bloque
	int num_bloques = (dimension+tamanio_bloque-1)/tamanio_bloque;
	//esto que acabamos de construir, nos va a sevir para dividir cualquier vector o arreglo para trabajar con cuda
	cudaEventRecord(inicio,0);	
	sumaVectores<<<num_bloques,tamanio_bloque>>>(A_dispositivo,B_dispositivo,C_dispositivo,dimension);
	//regresamos la informacion al host
	cudaEventRecord(final,0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(final);
	cudaEventElapsedTime(&tiempo,inicio,final);
	printf("\nEl tiempo total es: %f\n",tiempo/1000);
	cudaMemcpy(C_host,C_dispositivo, dimension*sizeof(int),cudaMemcpyDeviceToHost);
	/*printf("\nEl arreglo C es:\n");
	for(int i=0;i<dimension;++i)
		printf("%d ",C_host[i]);
	printf("\n");
	*/
	cudaFree(A_dispositivo);
	cudaFree(B_dispositivo);
	cudaFree(C_dispositivo);
	free(A_host);
	free(B_host);
	free(C_host);
}