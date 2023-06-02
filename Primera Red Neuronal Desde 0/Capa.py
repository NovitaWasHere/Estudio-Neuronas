import numpy as np
import math

from scipy import stats

class capa():
    def __init__(selft,n_neuronas_capa_anterior,n_neuronas,funcion_act):
        self.funcion_act = funcion_act
        #Truncnorm una funcion que permite crear numero aleatorios dado un rango, media y desviación estándar
        self.b =  np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)
        #- 1 = valor minimo , 1 = valor Maximo , loc = valor media , scale = desviacion estandar , size = tamaño de la muestra
        # reshape(1, n_neuronas): Esto reformatea el arreglo resultante en una matriz de dimensiones 1 x n_neuronas. La matriz tendrá una sola fila y n_neuronas columnas.
        self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)
