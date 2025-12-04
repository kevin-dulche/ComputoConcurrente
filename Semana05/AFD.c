#include <stdio.h>
#include <string.h>

typedef enum { 
    S0, 
    S1, 
    S2, 
    S3, 
    S4, 
    Sf, 
    DUMP 
} Estado;

Estado procesarComentario(const char *cadena) {
    Estado estado = S0;
    int i = 0;

    while (cadena[i] != '\0') {
        char c = cadena[i];

        switch (estado) {
            case S0:
                if (c == '/') estado = S1;
                else estado = DUMP;
                break;

            case S1:
                if (c == '/') estado = S2;  // Comentario de una línea
                else if (c == '*') estado = S3;  // Comentario multilínea
                else estado = DUMP;
                break;

            case S2:  // Comentario de una línea, sigue hasta '\n' o fin de cadena
                if (c == '\n') estado = Sf;
                break;

            case S3:  // Comentario multilínea, busca '*'
                if (c == '*') estado = S4;
                break;

            case S4:  // Se encontró '*', verificar si sigue '/'
                if (c == '/') estado = Sf;  // Comentario válido
                else if (c != '*') estado = S3;  // Regresar a q3 si no es '/'
                break;

            case Sf:
                return Sf;  // Comentario válido, terminamos

            case DUMP:
                return DUMP;  // Cadena no válida
        }

        i++;
    }

    // Si terminó en un estado válido, se acepta
    return (estado == Sf || estado == S2) ? Sf : DUMP;
}

int main() {
    char entrada[256];

    printf("Ingrese un comentario en C: ");
    fgets(entrada, sizeof(entrada), stdin);

    // Eliminar el salto de línea del final si lo tiene
    entrada[strcspn(entrada, "\n")] = 0;

    if (procesarComentario(entrada) == Sf) {
        printf("Comentario válido en C.\n");
    } else {
        printf("Entrada inválida.\n");
    }

    return 0;
}