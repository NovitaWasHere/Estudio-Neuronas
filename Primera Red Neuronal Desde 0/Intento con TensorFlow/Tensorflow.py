#dataset Fashion MNIST
#Se utilizará la biblioteca de 10.000 imagenes con etiquetas ya puestas para entrenar a la neurona
# from tensorflow.keras.datasets import fashion_mnist
# (X, y), (X_test, y_test) = fashion_mnist.load_data()

# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

# Dividimos el dataset en dos partes 10% de las imagenes totales serán para testeo y el 90% restante para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
print("Imágenes de entrenamiento", X_train.shape)
print("Imágenes de test", X_test.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# Configuramos como se entrenará la red
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers(learning_rate=0.001),
    metrics=["accuracy"]
)

# Definimos los parametros de entrenamiento
params = {
    "validation_data": (X_val,y_val),
    "epochs": 100,
    "verbose": 2,
    "batch_size":256,
}

# Iniciamos el entrenamiento
model.fit(X_train,y_train,**params)
