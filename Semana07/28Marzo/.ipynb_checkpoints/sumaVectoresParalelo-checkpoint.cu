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

void asignarMemoria(int **A, int **B, int **C, int dimension){
	printf("Asignando memoria para el arreglo A\n");
	*A = (int *)malloc(dimension*sizeof(int));
	if (*A==NULL){
		printf("Error al asignar memoria para A\n");
		exit(1);
	}
	
	printf("Asignando memoria para el arreglo B\n");
	*B = (int *)malloc(dimension*sizeof(int));
	if (*B==NULL){
		printf("Error al asignar memoria para B\n");
		free(*A);
		exit(1);
	}

	printf("Asignando memoria para el arreglo C\n");
	*C = (int *)malloc(dimension*sizeof(int));
	if (*C==NULL){
		printf("Error al asignar memoria para C\n");
		free(*A);
		free(*B);
		exit(1);
	}
}


void inicializarArreglos(int *A, int *B, int *C, int dimension){
	printf("Inicializando los arreglos A y B con números aleatorios\n");
	for(int i=0;i<dimension;++i){
		A[i]= 10 + rand() % 90;
		B[i]= 10 + rand() % 90;
	}
}


int main(int argc, char * argv[]){

	if(argc!=3){
		printf("Uso: %s <dimension> <numero de hilos por bloque>\n",argv[0]);
		exit(1);
	}
	
	srand(time(NULL));
	int dimension=atoi(argv[1]);//el tamaño que tendran los arreglos

	if(dimension<=0){
		printf("La dimension debe ser mayor a 0\n");
		exit(1);
	}

	int numero_hilos=atoi(argv[2]);
	if(numero_hilos<=0){
		printf("El numero de hilos por bloque debe ser mayor a 0\n");
		exit(1);
	}
	if(numero_hilos>1024){
		printf("El numero de hilos por bloque no puede ser mayor a 1024\n");
		exit(1);
	}

	cudaEvent_t inicio, final;
	cudaEventCreate(&inicio);
	cudaEventCreate(&final);
	float tiempo;
	int *A_dispositivo, *B_dispositivo, *C_dispositivo;
	int *A_host, *B_host, *C_host;
	//TENGO EL MANEJO DE LOS ARREGLOS EN EL HOST
	asignarMemoria(&A_host,&B_host,&C_host,dimension);
	
	inicializarArreglos(A_host,B_host,C_host,dimension);

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
	//int tamanio_bloque = propiedades.maxThreadsPerBlock; //numero de hilos por bloque
	
	int tamanio_bloque = numero_hilos; //numero de hilos por bloque
	int num_bloques = (dimension+tamanio_bloque-1)/tamanio_bloque;
	//esto que acabamos de construir, nos va a sevir para dividir cualquier vector o arreglo para trabajar con cuda
	cudaEventRecord(inicio,0);	
	printf("Sumando los elementos de los arreglos A y B\n");
	sumaVectores<<<num_bloques,tamanio_bloque>>>(A_dispositivo,B_dispositivo,C_dispositivo,dimension);
	//regresamos la informacion al host
	cudaDeviceSynchronize();
	cudaEventRecord(final,0);
	cudaEventSynchronize(final);
	cudaEventElapsedTime(&tiempo,inicio,final);
	cudaMemcpy(C_host,C_dispositivo, dimension*sizeof(int),cudaMemcpyDeviceToHost);
	printf("El tiempo de ejecución es de: %f segundos\n",tiempo/1000);
	
	cudaFree(A_dispositivo);
	cudaFree(B_dispositivo);
	cudaFree(C_dispositivo);

	free(A_host);
	free(B_host);
	free(C_host);

	return 0;
}