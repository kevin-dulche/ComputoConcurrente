#include <stdio.h>
#include <string.h>

typedef enum { 
    S0, 
    S1, 
    S2, 
    S3, 
    S4, 
    S5
} Estado;

Estado procesarComentario(const char *cadena) {
    Estado estado = S0;

    int i = 0;

    while (cadena[i] != '\0') {
        char c = cadena[i];

        switch (estado) {
            case S0:
                if (c == 'a') estado = S1;
                else if (c == 'b') estado = S3;
                else if (c == 'c') estado = S3;
                break;

            case S1:
                if (c == 'a') estado = S4;
                else if (c == 'b') estado = S5;
                else if (c == 'c') estado = S2;
                break;

            case S2:
                estado = S2;
                break;

            case S3:
                if (c == 'a') estado = S4;
                else if (c == 'b') estado = S3;
                else if (c == 'c') estado = S3;
                break;

            case S4:
                if (c == 'a') estado = S4;
                else if (c == 'b') estado = S5;
                else if (c == 'c') estado = S3;
                break;

            case S5:
                if (c == 'a') estado = S4;
                else if (c == 'b') estado = S3;
                else if (c == 'c') estado = S3;
                break;
        }

        i++;
    }

    // Si terminó en un estado válido, se acepta
    return (estado == S0 || estado == S1 || estado == S3 || estado == S4) ? 1 : 0;
}

int main() {
    char entrada[256];

    printf("La cadena con letras a,b,c: ");
    fgets(entrada, sizeof(entrada), stdin);

    // Eliminar el salto de línea del final si lo tiene
    entrada[strcspn(entrada, "\n")] = 0;

    if (procesarComentario(entrada) == 1) {
        printf("Cadena valida.\n");
    } else {
        printf("Entrada inválida.\n");
    }

    return 0;
}