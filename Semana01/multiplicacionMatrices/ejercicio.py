def pedirDimensiones(nMatriz:int) -> tuple:
    filas = int(input("Ingrese el número de filas de la matriz {}: ".format(nMatriz)))
    columnas = int(input("Ingrese el número de columnas de la matriz {}: ".format(nMatriz)))
    return filas, columnas

def llenarMatriz(filas:int, columnas:int, nMatriz:int) -> list:
    matriz = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            fila.append(int(input("Ingrese el valor de la posicion [{}][{}] de la matriz {}: ".format(i, j, nMatriz))))
        matriz.append(fila)
    return matriz

def calcularMultiplicacion(filasM1, columnasM1, filasM2, columnasM2, matriz1, matriz2) -> list:
    if columnasM1 != filasM2:
        return None
    matrizResultado = []
    for i in range(filasM1):
        fila = []
        for j in range(columnasM2):
            suma = 0
            for k in range(columnasM1):
                suma += matriz1[i][k] * matriz2[k][j]
            fila.append(suma)
        matrizResultado.append(fila)
    return matrizResultado

def imprimirMatriz(filas:int, columnas:int, matriz: list) -> None:
    for i in range(filas):
        for j in range(columnas):
            print(matriz[i][j], end=" ")
        print()
    print()


def main() -> None:
    print("Multiplicación de matrices")

    columnasM1 = 0
    filasM2 = 1
    while columnasM1 != filasM2:
        filasM1, columnasM1 = pedirDimensiones(1)
        filasM2, columnasM2 = pedirDimensiones(2)
        if columnasM1 != filasM2:
            print("El número de columnas de la matriz 1 debe ser igualal número de filas de la matriz 2")
    
    matriz1 = llenarMatriz(filasM1, columnasM1, "1")
    matriz2 = llenarMatriz(filasM2, columnasM2, "2")

    matrizResultado = calcularMultiplicacion(filasM1, columnasM1, filasM2, columnasM2, matriz1, matriz2)

    print("\nMatriz 1:")
    imprimirMatriz(filasM1, columnasM1, matriz1)

    print("Matriz 2:")
    imprimirMatriz(filasM2, columnasM2, matriz2)

    print("El resultado de la multiplicación de las matrices es:")
    imprimirMatriz(filasM1, columnasM2, matrizResultado)

if __name__ == "__main__":
    main()