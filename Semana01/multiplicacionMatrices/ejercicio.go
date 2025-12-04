package main

import "fmt"

func pedirDimensiones(filas *int, columnas *int, nMatriz int) {
    fmt.Printf("Ingresa el número de filas de la matriz %d: ", nMatriz)
    fmt.Scan(filas) // Lee y guarda en la dirección de memoria de 'filas'

    fmt.Printf("Ingresa el número de columnas de la matriz %d: ", nMatriz)
    fmt.Scan(columnas) // Lee y guarda en la dirección de memoria de 'columnas'
}

func pedirMatriz(filas int, columnas int, matriz [][]int) {
    for i := 0; i < filas; i++ {
        for j := 0; j < columnas; j++ {
            fmt.Printf("Ingresa el valor de la posición [%d][%d]: ", i, j)
            fmt.Scan(&matriz[i][j]) // Lee y guarda en la dirección de memoria de 'matriz[i][j]'
        }
    }
}

func llenarMatriz(filas int, columnas int, matriz [][]int, nMatriz int) {
    for i := 0; i < filas; i++ {
        for j := 0; j < columnas; j++ {
            fmt.Printf("Ingresa el valor de la posición [%d][%d] de la matriz %d: ", i, j, nMatriz)
            fmt.Scan(&matriz[i][j]) // Lee y guarda en la dirección de memoria de 'matriz[i][j]'
        }
    }
}

func calcularMultiplicacion(filasM1 int, columnasM1 int, filasM2 int, columnasM2 int, matriz1 [][]int, matriz2 [][]int, resultado [][]int) {
    for i := 0; i < filasM1; i++ {
        for j := 0; j < columnasM2; j++ {
            for k := 0; k < columnasM1; k++ {
                resultado[i][j] += matriz1[i][k] * matriz2[k][j]
            }
        }
    }
}

func imprimirMatriz(filas int, columnas int, matriz [][]int) {
    for i := 0; i < filas; i++ {
        for j := 0; j < columnas; j++ {
            fmt.Printf("%d ", matriz[i][j])
        }
        fmt.Println()
    }
    fmt.Println()
}

func crearMatriz(filas, columnas int) [][]int {
    matriz := make([][]int, filas) // Crea un slice de filas
    for i := range matriz {
        matriz[i] = make([]int, columnas) // Cada fila es un slice de columnas
    }
    return matriz
}

func main() {
    filasM1 := 0;
    columnasM1 := 0;

    filasM2 := 0;
    columnasM2 := 0;

    filasResultado := 0;
    columnasResultado := 0;

    fmt.Println("Multiplicacion de matrices");

    for {
        pedirDimensiones(&filasM1, &columnasM1, 1);
        pedirDimensiones(&filasM2, &columnasM2, 2);

        if columnasM1 != filasM2 {
            fmt.Println("El número de columnas de la matriz 1 debe ser igual al número de filas de la matriz 2");
        } else {
            break;
        }
    }

    filasResultado = filasM1;
    columnasResultado = columnasM2;

    matriz1 := crearMatriz(filasM1, columnasM1)
    matriz2 := crearMatriz(filasM2, columnasM2)

    llenarMatriz(filasM1, columnasM1, matriz1, 1);
    llenarMatriz(filasM2, columnasM2, matriz2, 2);

    resultado := crearMatriz(filasResultado, columnasResultado)

    calcularMultiplicacion(filasM1, columnasM1, filasM2, columnasM2, matriz1, matriz2, resultado);

    fmt.Println("\nMatriz 1:");
    imprimirMatriz(filasM1, columnasM1, matriz1);

    fmt.Println("Matriz 2:");
    imprimirMatriz(filasM2, columnasM2, matriz2);

    fmt.Println("El resultado de la multiplicación de las matrices es:");
    imprimirMatriz(filasResultado, columnasResultado, resultado);
}