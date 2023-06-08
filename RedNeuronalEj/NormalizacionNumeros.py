import numpy as np
from keras.models import Sequential

#Manera en la que se normalizan los datos, aunque a veces da fallos la desnormalizacion
def normalize_data(data):
    min_value = np.min(data)
    max_value = np.max(data)
    range = max_value - min_value
    normalized_data = (data - min_value) / range
    return normalized_data
# Ejemplo de uso
training_data = np.array([1, 2, 3, 4, 5])
target_data = np.array([6, 7, 8, 9, 10])
normalized_training_data = normalize_data(training_data)
normalized_target_data = normalize_data(target_data)
print(normalized_training_data)
print(normalized_target_data)
def denormalize_data(normalized_data, original_data):
    min_value = np.min(original_data)
    max_value = np.max(original_data)
    range = max_value - min_value
    denormalized_data = (normalized_data * range) + min_value
    return denormalized_data
training_data = denormalize_data(normalized_training_data, training_data)
target_data = denormalize_data(normalized_target_data, target_data)
print(training_data)
print(target_data)

#La manera en la que no da fallo la desnormalizacion de los datos es la siguiente

# Normalizamos los datos de entrada y salida en el rango [0, 1]
input_number = float(input("Numero a dar"))
model = Sequential()
training_min = np.min(training_data)
training_max = np.max(training_data)
training_data = (training_data - training_min) / (training_max - training_min)

target_min = np.min(target_data)
target_max = np.max(target_data)
target_data = (target_data - target_min) / (target_max - target_min)

normalized_number = (input_number - training_min) / (training_max - training_min)
prediction = model.predict(np.array([[normalized_number]]))
output_number = (prediction * (target_max - target_min)) + target_min

print("El resultado de duplicar", input_number, "es el siguiente = ", output_number[0])