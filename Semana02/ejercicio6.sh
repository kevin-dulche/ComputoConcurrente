# !/bin/bash
# Programa de prueba para el ejercico 6

gcc -o ejercicio6 ejercicio6.c

for i in $(seq 10 100); do
    ./ejercicio6 $i
done

rm ejercicio6

echo "Fin del programa"