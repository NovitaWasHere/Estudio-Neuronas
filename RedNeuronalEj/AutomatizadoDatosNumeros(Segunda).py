import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense
np.set_printoptions(suppress=True)

array_numeros = np.array([], "float32")  # Inicializar el arreglo vacío
training_data = np.array([], "float32")
for x in range(1, 10001):
    numero = x
    array_numeros = np.append(array_numeros, numero)

print(array_numeros)
print(array_numeros*2)
# Cargamos los datos de entrenamiento
training_data = np.append(training_data, array_numeros)
training_data = training_data.reshape(-1, 1)  # Reshape a matriz de columna

# Cargamos los resultados esperados
target_data = np.array([], "float32")
target_data = np.append(target_data, array_numeros*2)
target_data = target_data .reshape(-1, 1)


# Normalizamos los datos de entrada y salida en el rango [0, 1]
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
target_data = scaler.transform(target_data)

# Creamos el modelo de red neuronal secuencial
model = Sequential()

model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(68, activation='relu'))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation='linear'))

# Compilamos el modelo con la función de pérdida y el optimizador
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])

# Entrenamos el modelo con los datos de entrenamiento
model.fit(training_data, target_data, epochs=300)

# Evaluamos el modelo con los mismos datos de entrenamiento
scores = model.evaluate(training_data, target_data)

# Desnormalizamos los datos de entrenamiento y resultado esperado
training_data = scaler.inverse_transform(training_data)
target_data = scaler.inverse_transform(target_data)

# Predecimos con el modelo el resultado de duplicar un número específico
input_number = 26
normalized_number = scaler.transform([[input_number]])
prediction = model.predict(normalized_number)
output_number = scaler.inverse_transform(prediction)

# Imprimimos los resultados
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("El resultado de duplicar", input_number, "es", output_number[0, 0])
print("Predicciones:")
print(scaler.inverse_transform(model.predict(training_data)).round())
