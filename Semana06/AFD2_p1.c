// Ejercicio 2: Implementar un AFD que acepte cadenas que contengan un número par de a's y un número par de b's.

#include <stdio.h>
#include <string.h>

typedef enum { 
    S0, 
    S1, 
    S2, 
    S3
} Estado;

Estado procesarComentario(const char *cadena) {
    Estado estado = S0;
    int i = 0;

    while (cadena[i] != '\0') {
        char c = cadena[i];

        switch (estado) {
            case S0:
                if (c == 'a') estado = S1;
                else estado = S2;
                break;

            case S1:
                if (c == 'a') estado = S0;
                else estado = S3; 
                break;

            case S2:
                if (c == 'a') estado = S3;
                else estado = S0;
                break;

            case S3: 
                if (c == 'a') estado = S2;
                else estado = S1;
                break;
        }

        i++;
    }

    return (estado == S0 || estado == S1 || estado == S2) ? 1 : 0;
}

int main() {
    char entrada[256];

    printf("Ingrese una cadena con a's y b's -> ");
    fgets(entrada, sizeof(entrada), stdin);

    // Eliminar el salto de línea del final si lo tiene
    entrada[strcspn(entrada, "\n")] = 0;

    if (procesarComentario(entrada) == 1) {
        printf("Cadena valida\n");
    } else {
        printf("Cadena inválida.\n");
    }

    return 0;
}