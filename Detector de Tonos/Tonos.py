import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Cargamos los datos de entrenamiento
training_data = np.array([], "float32")
# Cargamos los resultados esperados
target_data = np.array([], "float32")

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






