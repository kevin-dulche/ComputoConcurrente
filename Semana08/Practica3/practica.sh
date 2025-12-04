# !/bin/bash

echo "Compilando los programas..."
gcc sumaSecuencial.c -o sS
gcc CHAT.c -o sC
nvcc sumaParalela.cu -o sP

echo "Ejecutando los programas..."
echo "Ejecutando sumaSecuencial..."
./sS 10000
./sS 10000
./sS 10000
./sS 10000
echo "Ejecutando CHAT..."
./sC 10000 2
./sC 10000 4
./sC 10000 8
./sC 10000 16
echo "Ejecutando sumaParalela..."
./sP 10000
./sP 10000
./sP 10000
./sP 10000