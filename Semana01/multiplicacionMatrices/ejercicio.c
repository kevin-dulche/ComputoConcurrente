#include <stdio.h>

void pedirDimensiones(int *filas, int *columnas, int nMatriz){
    printf("Ingresa el número de filas de la matriz %d: ", nMatriz);
    scanf("%d", filas);

    printf("Ingresa el número de columnasde la matriz %d: ", nMatriz);
    scanf("%d", columnas);
}

void llenarMatriz(int filas, int columnas, int matriz[filas][columnas], int nMatriz){
    for (int i = 0; i < filas; i++){
        for (int j = 0; j < columnas; j++){
            printf("Ingresa el valor de la posición [%d][%d] de la matriz %d: ", i, j, nMatriz);
            scanf("%d", &matriz[i][j]);
        }
    }
}

void calcularMultiplicacion(int filasM1, int columnasM1, int matriz1[filasM1][columnasM1], int filasM2, int columnasM2, int matriz2[filasM2][columnasM2], int resultado[filasM1][columnasM2]){
    for (int i = 0; i < filasM1; i++){
        for (int j = 0; j < columnasM2; j++){
            resultado[i][j] = 0;
            for (int k = 0; k < columnasM1; k++){
                resultado[i][j] += matriz1[i][k] * matriz2[k][j];
            }
        }
    }
}

void imprimirMatriz(int filas, int columnas, int matriz[filas][columnas]){
    for (int i = 0; i < filas; i++){
        for (int j = 0; j < columnas; j++){
            printf("%d ", matriz[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char const *argv[]){
    int filasM1 = 0;
    int columnasM1 = 0;

    int filasM2 = 0;
    int columnasM2 = 0;

    int filasResultado = 0;
    int columnasResultado = 0;

    printf("Multiplicación de matrices\n");

    do {
        pedirDimensiones(&filasM1, &columnasM1, 1);
        pedirDimensiones(&filasM2, &columnasM2, 2);
        if (columnasM1 != filasM2){
            printf("El número de columnas de la matriz 1 debe ser igual al número de filas de la matriz 2\n");
        }
    } while (columnasM1 != filasM2);

    filasResultado = filasM1;
    columnasResultado = columnasM2;

    int matriz1[filasM1][columnasM1];
    int matriz2[filasM2][columnasM2];

    llenarMatriz(filasM1, columnasM1, matriz1, 1);
    llenarMatriz(filasM2, columnasM2, matriz2, 2);

    int resultado[filasResultado][columnasResultado];

    calcularMultiplicacion(filasM1, columnasM1, matriz1, filasM2, columnasM2, matriz2, resultado);

    printf("\nMatriz 1:\n");
    imprimirMatriz(filasM1, columnasM1, matriz1);

    printf("Matriz 2:\n");
    imprimirMatriz(filasM2, columnasM2, matriz2);

    printf("El resultado de la multiplicación de las matrices es:\n");
    imprimirMatriz(filasResultado, columnasResultado, resultado);

    return 0;
}