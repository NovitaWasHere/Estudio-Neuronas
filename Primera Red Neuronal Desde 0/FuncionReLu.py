import numpy as np
import math
import matplotlib.pyplot as plt

def derivada_relu(x):
  # Si elementos de x son menores o iguales a cero el valor será igual a cero 
  x[x<=0] = 0
  # Si elementos de x son mayores a cero el valor será igual a uno
  x[x>0] = 1
  return x

relu = (
  lambda x: x * (x > 0),
  lambda x:derivada_relu(x)
  )
# Volvemos a definir rango que ha sido cambiado
rango = np.linspace(-10,10).reshape([50,1])

#Esto multiplica cada elemento del arreglo rango por uno si es mayor a cero y por cero si es menor o igual a cero
datos_relu = relu[0](rango)
#Esto aplica la función derivada_relu al arreglo rango para obtener la derivada en cada punto.
datos_relu_derivada = relu[1](rango)

# Se limpia cualquier gráfico existente en la figura actual 
plt.cla()
# La figura tiene una fila y dos columnas, y tiene un tamaño de 15 pulgadas de ancho y 5 pulgadas de alto.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
axes[0].plot(rango, datos_relu[:,0])
axes[1].plot(rango, datos_relu_derivada[:,0])
plt.show()