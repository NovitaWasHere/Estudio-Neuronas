import tensorflow as tf
import numpy as np

# Datos de entrenamiento
X_train = np.array([[2], [4], [6], [8], [10]])  # Entradas
y_train = np.array([[4], [8], [12], [16], [20]])  # Salidas esperadas

# Construir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compilar el modelo
model.compile(optimizer='sgd', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Obtener entrada del usuario
input_num = float(input("Ingrese un número: "))

# Preparar dato de prueba
X_test = np.array([[input_num]])

# Realizar predicción
y_pred_test = model.predict(X_test)

# Imprimir resultado
print("El número duplicado es:", y_pred_test[0][0])