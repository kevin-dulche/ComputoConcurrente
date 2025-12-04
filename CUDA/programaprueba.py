import cupy as cp

# Obtener el nombre de la primera GPU
print(cp.cuda.get_device_name(0))