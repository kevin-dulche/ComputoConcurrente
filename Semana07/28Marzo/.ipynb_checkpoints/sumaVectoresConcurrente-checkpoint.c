
//Vamos a hacer un programa que realice la suma de los elementos de dos vectores

//nuestra primera version (en C o C++) sigue la taxonomía de MIMD (ejecutan los nucleos mediante los hilos un conjunto de instrucciones en diferentes datos de forma simultanea (concurrente)

//la segunda version (que haremos el miercoles) en CUDA, sigue la taxonomia SIMD porque el mismo conjunto de instrucciones se va a ejectuar "al mismo tiempo"
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

typedef struct{
	float * A;
	float * B;
	int inicio;
	int fin;
	int id;
}parametros;


void * suma_vectores(void * arg){
	parametros * argumentos = (parametros *)arg;
	//printf("\nSoy el hilo %d y voy a comenzar a sumar mis elementos\n", argumentos->id);
	for(int i = argumentos->inicio;i < argumentos ->fin; ++i)
		argumentos->A[i]+=argumentos->B[i];//realizamos la suma de los elementos correspondientes y los almacenamos en A
		// printf("\nSoy el hilo %d y termine de sumar mis elementos\n", argumentos->id);
	pthread_exit(NULL);
}


void asignarMemoria(float **A, float **B, float **C, int n)
{
	printf("Asignando memoria para el arreglo A\n");
	*A = (float *)malloc(n * sizeof(float));
	if (*A == NULL)
	{
		printf("Error al asignar memoria para A\n");
		exit(1);
	}
	printf("Asignando memoria para el arreglo B\n");
	*B = (float *)malloc(n * sizeof(float));
	if (*B == NULL)
	{
		printf("Error al asignar memoria para B\n");
		free(*A);
		exit(1);
	}
	printf("Asignando memoria para el arreglo C\n");
	*C = (float *)malloc(n * sizeof(float));
	if (*C == NULL)
	{
		printf("Error al asignar memoria para C\n");
		free(*A);
		free(*B);
		exit(1);
	}
}


void inicializarArreglos(float *A, float *B, int n)
{
	srand(time(NULL));
	printf("Inicializando los arreglos A y B con números aleatorios\n");
	for (int i = 0; i < n; i++)
	{
		A[i] = (float)rand() / RAND_MAX;
		B[i] = (float)rand() / RAND_MAX;
	}
}


int main(int argc, char * argv[]){
	if(argc != 3){
		printf("Uso: %s <tamanio del vector> <numero de hilos>\n", argv[0]);
		exit(1);
	}

	srand(time(NULL));
	int N=atoi(argv[1]);
	int num_hilos=atoi(argv[2]);
	int tamanioBloque = N / num_hilos; //tenemos que cada bloque es del tamaño correspondiente a la proporcion que le toca a cada hilo
	int sobrante =  N % num_hilos;
	struct timeval start, end;
	/*ojo... hay que verificar que los valores sean divididos de tal manera que no perdamos informacion*/
	float *A, *B, *C;//trabajamos con memoria dinamica
	//dado que queremos que sean bastante grandes
	asignarMemoria(&A, &B, &C, N);

	//el arreglito de argumentos para cada hilo
	parametros argumentos[num_hilos];
	
	inicializarArreglos(A, B, N);
	
	// inicializamos el tiempo
	pthread_t hilos[num_hilos];
	gettimeofday(&start, NULL);
	printf("Sumando los elementos de los arreglos A y B\n");
	for(int i=0;i<num_hilos;++i){
		argumentos[i].A = A;
		argumentos[i].B = B;
		argumentos[i].inicio = i * tamanioBloque;
		argumentos[i].fin = (i + 1)* tamanioBloque;
		argumentos[i].id = i+1;
		if(i == num_hilos - 1)//si es el ultimo hilo
			argumentos[i].fin = N;
		
		pthread_create(&hilos[i],NULL,suma_vectores, (void *) &argumentos[i]);
	}

	for (int i=0;i<num_hilos;++i){
		pthread_join(hilos[i],NULL);
	}

	gettimeofday(&end, NULL);  // Fin después de que terminen los hilos

    double tiempo_concurrente = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Tiempo de ejecución es de: %f segundos\n", tiempo_concurrente);
	//opcional si es windows su s.o.
	free(A);
	free(B);
}
