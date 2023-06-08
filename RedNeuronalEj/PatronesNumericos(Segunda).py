import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Cargamos los datos de entrenamiento
training_data = np.array([[-2.0], [0], [1], [2.0], [3.0], [4.0], [5.0], [6.0], [7.2], [8.4], [9.0], [10.0],
                          [11.0], [12.0], [13.0], [14.5], [15.0], [16.0], [17.0], [18.0], [19.0], [20.7],
                          [25.0], [30.0], [35.0], [40.0], [45.0], [50.0], [55.0], [70.5], [100.4],
                          [120.0], [140.0], [180.0], [200.0], [400.0], [720.3], [1000.0], [1100.4], [1120.0],
                          [1140.0], [1180.0], [1200.0], [1400.0], [1720.3], [2000.0],
                          [3140.0], [3180.0], [3200.0], [3400.0], [3720.3], [4000.0],
                          [5140.0], [5180.0], [5200.0], [5400.0], [5720.3], [6000.0],
                          [7140.0], [7180.0], [7200.0], [7400.0], [7720.3], [8000.0]], "float32")
# Cargamos los resultados esperados
target_data = np.array([[-4.0], [0], [2], [4.0], [6.0], [8.0], [10.0], [12.0], [14.4], [16.8], [18.0], [20.0],
                        [22.0], [24.0], [26.0], [29.0], [30.0], [32.0], [34.0], [36.0], [38.0], [41.4],
                        [50.0], [60.0], [70.0], [80.0], [90.0], [100.0], [110.0], [141.0], [200.8],
                        [240.0], [280.0], [360.0], [400.0], [800.0], [1440.6], [2000.0], [2200.8], [2240.0],
                        [2280.0], [2360.0], [2400.0], [2800.0], [3440.6], [4000.0],
                        [6280.0], [6360.0], [6400.0], [6800.0], [7440.6], [8000.0],
                        [10280.0], [10360.0], [10400.0], [10800.0], [11440.6], [12000.0],
                        [14280.0], [14360.0], [14400.0], [14800.0], [15440.6], [16000.0]], "float32")

def normalize_data(data):
    min_value = np.min(data)
    max_value = np.max(data)
    range_value = max_value - min_value
    # Comprobamos si el rango es cero
    if range_value == 0:
        # Si el rango es cero, establecemos los datos normalizados en 0
        normalized_data = np.zeros_like(data)
    else:
        # De lo contrario, realizamos la normalización
        normalized_data = (data - min_value) / range_value
    return normalized_data

def denormalize_data(normalized_data, original_data):
    min_value = np.min(original_data)
    max_value = np.max(original_data)
    range_value = max_value - min_value
    denormalized_data = (normalized_data * range_value) + min_value
    return denormalized_data

training_data = normalize_data(training_data)
target_data = normalize_data(target_data)

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
model.fit(training_data, target_data, epochs=5000)

# Evaluamos el modelo con los mismos datos de entrenamiento
scores = model.evaluate(training_data, target_data)

training_data = denormalize_data(training_data, training_data)
target_data = denormalize_data(target_data, target_data)

# Predecimos con el modelo el resultado de duplicar un número específico
input_number = float(input("Introduce el valor a duplicar: "))
normalized_number = normalize_data(np.array([[input_number]], dtype="float32"))
prediction = model.predict(normalized_number)
output_number = denormalize_data(prediction, prediction)

# Imprimimos los resultados
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("El resultado de duplicar", input_number, "es el siguiente = ", output_number[0])
print("Predicciones posteriores:")
print(denormalize_data(model.predict(training_data).round(), target_data))
