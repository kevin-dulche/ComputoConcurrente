// Ejercicio 3: Programa que acepte nombres de identificadores (los nombres de las variables de un programa). Los identificadores aceptados deberán ser cadenas que empiezan con letras minúsculas del alfabeto inglés [a − z] o con un guión bajo, exclusivamente. Las cadenas podrán contener además, letras [a − z] y dígitos [0 − 9].

#include <stdio.h>
#include <string.h>

typedef enum { 
    S0, 
    S1, 
    D 
} Estado;

Estado procesarComentario(const char *cadena) {
    Estado estado = S0;
    int i = 0;

    while (cadena[i] != '\0') {
        char c = cadena[i];

        switch (estado) {
            case S0:
                if ((c >= 'a' && c <= 'z') || c == '_') estado = S1;
                else estado = D;
                break;

            case S1:
                return 1;
                break;

            case D:
                return 0;
                break;
        }

        i++;
    }

    return (estado == S1) ? 1 : 0;
}

int main() {
    char entrada[256];

    printf("Ingrese el nombre de una variable -> ");
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