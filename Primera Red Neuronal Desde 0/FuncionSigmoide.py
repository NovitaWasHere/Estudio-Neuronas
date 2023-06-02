import numpy as np
import math
import matplotlib.pyplot as plt


sigmoid = (
  
  #Función sigmoide toma un valor x utiliza la fórmula 1 / (1 + e^(-x)), donde np.exp es la base del logaritmo natural. 
  lambda x:1 / (1 + np.exp(-x)),
  #Función es la derivada de la función sigmoide, para calcular el gradiente durante el entrenamiento de modelos de aprendizaje automático.
  lambda x:x * (1 - x)

)
#Se crea un rango de -10 a 10 utilizando la función linspace de NumPy, y luego se da forma a una matriz de tamaño 50x1.
rango = np.linspace(-10,10).reshape([50,1])
#Se calculan los valores de la función sigmoide y su derivada para el rango de valores utilizando las funciones definidas anteriormente.
datos_sigmoide = sigmoid[0](rango)
datos_sigmoide_derivada = sigmoid[1](rango)

#Se crea una figura con dos subgráficos utilizando subplots de matplotlib:
#Esto crea una figura con 1 fila y 2 columnas de subgráficos. figsize=(15, 5) establece el tamaño de la figura en pulgadas.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
#Se traza el gráfico de la función sigmoide en el primer subgráfico:
axes[0].plot(rango, datos_sigmoide)
#Se traza el gráfico de la derivada de la función sigmoide en el segundo subgráfico:
axes[1].plot(rango, datos_sigmoide_derivada)
#Se ajusta el diseño de los gráficos para que se ajusten correctamente en la figura:
fig.tight_layout()