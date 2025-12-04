#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>//para memoria dinamica, conversiones, etc... en este caso la usamos para atoi
#include <time.h>

#define inicio 1
#define meta 30

int ganador=-1;
//[nombre_programa, argv[1], argv[2]]
void * brinca(void * arg){
	int * id = (int *) arg;
	int posicion=inicio;
	while(posicion<meta && ganador == -1){
        posicion += rand() % 10 + 1;//valores entre 1 y 10
        printf("\nSoy la ranita %d y estoy en la posicion %d",*id,posicion);
        sleep(1);
	}

	if(posicion>=meta&&ganador==-1){
		ganador=*id;
		pthread_exit(NULL);
	}
}

int main(int argc, char * argv []){//vamos a trabajar con parametros del main
	
	int ranas = atoi(argv[1]);
    srand(time(NULL));//srand es de seed random time y NULL es que tomamos la hora de ejecucion del programa como semilla
	printf("\nVoy a mandar a competir a %d ranas\n",ranas);
	
    pthread_t ranitas[ranas];//creamos un arreglo de hilos (pthread_t)

	//vamos a crear los hilos
	for(int i=0;i<ranas;++i){
		int * id = malloc(sizeof(int));//cada ranita tiene su identificador "unico"
		* id = i;
		pthread_create(&ranitas[i],NULL, brinca,(void *)id);
	}
	
	//mandamos a esperar al main hasta que terminen los hilos
	for(int i=0;i<ranas;++i)
		pthread_join(ranitas[i],NULL);
	
	printf("\nLa ranita ganadora es: %d", ganador);
	return 0;
}