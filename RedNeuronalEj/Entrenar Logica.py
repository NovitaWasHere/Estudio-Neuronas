from keras.models import model_from_json
import numpy as np

# Cargar el modelo desde el archivo JSON y los pesos desde el archivo HDF5
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')

# Compilar el modelo cargado
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

numero = float(input("Introduce el valor correspondiente"))

# Datos de entrada para la predicción
input_data = np.array([[numero]], dtype=np.float32)

# Realizar la predicción
prediction = loaded_model.predict(input_data)

# Imprimir la predicción
print("Predicción:", prediction)
