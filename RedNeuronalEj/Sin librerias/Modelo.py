import numpy as np

# Cargamos los datos de entrenamiento
training_data = np.array([[-2.0], [0], [1], [2.0], [3.0], [4.0], [5.0], [6.0], [7.2], [8.4], [9.0], [10.0],
                          [11.0], [12.0], [13.0], [14.5], [15.0], [16.0], [17.0], [18.0], [19.0], [20.7],
                          [25.0], [30.0], [35.0], [40.0], [45.0], [50.0], [55.0], [70.5], [100.4],
                          [120.0], [140.0], [180.0], [200.0], [400.0], [720.3], [1000.0], [1100.4], [1120.0],
                          [1140.0], [1180.0], [1200.0], [1400.0], [1720.3], [2000.0],
                          [3140.0], [3180.0], [3200.0], [3400.0], [3720.3], [4000.0],
                          [5140.0], [5180.0], [5200.0], [5400.0], [5720.3], [6000.0],
                          [7140.0], [7180.0], [7200.0], [7400.0], [7720.3], [8000.0]], dtype="float32")
# Cargamos los resultados esperados
target_data = np.array([[-4.0], [0], [2], [4.0], [6.0], [8.0], [10.0], [12.0], [14.4], [16.8], [18.0], [20.0],
                        [22.0], [24.0], [26.0], [29.0], [30.0], [32.0], [34.0], [36.0], [38.0], [41.4],
                        [50.0], [60.0], [70.0], [80.0], [90.0], [100.0], [110.0], [141.0], [200.8],
                        [240.0], [280.0], [360.0], [400.0], [800.0], [1440.6], [2000.0], [2200.8], [2240.0],
                        [2280.0], [2360.0], [2400.0], [2800.0], [3440.6], [4000.0],
                        [6280.0], [6360.0], [6400.0], [6800.0], [7440.6], [8000.0],
                        [10280.0], [10360.0], [10400.0], [10800.0], [11440.6], [12000.0],
                        [14280.0], [14360.0], [14400.0], [14800.0], [15440.6], [16000.0]], dtype="float32")

# Normalizamos los datos de entrada y salida en el rango [0, 1]
training_min = np.min(training_data)
training_max = np.max(training_data)
training_data = (training_data - training_min) / (training_max - training_min)

target_min = np.min(target_data)
target_max = np.max(target_data)
target_data = (target_data - target_min) / (target_max - target_min)

# Inicializamos los pesos y los sesgos de manera aleatoria
np.random.seed(0)
weights1 = 0.1 * np.random.randn(1, 128)  # Pesos de la primera capa oculta (forma: (1, 128))
bias1 = np.zeros((1, 128))                # Sesgos de la primera capa oculta (forma: (1, 128))

weights2 = 0.1 * np.random.randn(128, 68) # Pesos de la segunda capa oculta (forma: (128, 68))
bias2 = np.zeros((1, 68))                  # Sesgos de la segunda capa oculta (forma: (1, 68))

weights3 = 0.1 * np.random.randn(68, 32)  # Pesos de la tercera capa oculta (forma: (68, 32))
bias3 = np.zeros((1, 32))                  # Sesgos de la tercera capa oculta (forma: (1, 32))

weights4 = 0.1 * np.random.randn(32, 16)  # Pesos de la cuarta capa oculta (forma: (32, 16))
bias4 = np.zeros((1, 16))                  # Sesgos de la cuarta capa oculta (forma: (1, 16))

weights5 = 0.1 * np.random.randn(16, 8)   # Pesos de la quinta capa oculta (forma: (16, 8))
bias5 = np.zeros((1, 8))                   # Sesgos de la quinta capa oculta (forma: (1, 8))

weights6 = 0.1 * np.random.randn(8, 1)    # Pesos de la capa de salida (forma: (8, 1))
bias6 = np.zeros((1, 1))                   # Sesgos de la capa de salida (forma: (1, 1))

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

# Función de activación lineal
def linear(x):
    return x

# Entrenamiento del modelo
epochs = 1000
learning_rate = 0.001

for epoch in range(epochs):
    # Forward propagation
    #Se realiza la multiplicacion de matrices y vectores con np.dot y se le suma el cesgo
    hidden1 = relu(np.dot(training_data, weights1) + bias1)
    hidden2 = relu(np.dot(hidden1, weights2) + bias2)
    hidden3 = relu(np.dot(hidden2, weights3) + bias3)
    hidden4 = relu(np.dot(hidden3, weights4) + bias4)
    hidden5 = relu(np.dot(hidden4, weights5) + bias5)
    output = linear(np.dot(hidden5, weights6) + bias6)

    # Cálculo del error
    error = output - target_data

    # Backpropagation
    gradient_output = error
    gradient_hidden5 = np.dot(gradient_output, weights6.T)
    gradient_hidden4 = np.dot(gradient_hidden5, weights5.T)
    gradient_hidden3 = np.dot(gradient_hidden4, weights4.T)
    gradient_hidden2 = np.dot(gradient_hidden3, weights3.T)
    gradient_hidden1 = np.dot(gradient_hidden2, weights2.T)

    # Actualización de pesos y sesgos
    weights6 -= learning_rate * np.dot(hidden5.T, gradient_output)
    bias6 -= learning_rate * np.sum(gradient_output, axis=0, keepdims=True)
    weights5 -= learning_rate * np.dot(hidden4.T, gradient_hidden5)
    bias5 -= learning_rate * np.sum(gradient_hidden5, axis=0, keepdims=True)
    weights4 -= learning_rate * np.dot(hidden3.T, gradient_hidden4)
    bias4 -= learning_rate * np.sum(gradient_hidden4, axis=0, keepdims=True)
    weights3 -= learning_rate * np.dot(hidden2.T, gradient_hidden3)
    bias3 -= learning_rate * np.sum(gradient_hidden3, axis=0, keepdims=True)
    weights2 -= learning_rate * np.dot(hidden1.T, gradient_hidden2)
    bias2 -= learning_rate * np.sum(gradient_hidden2, axis=0, keepdims=True)
    weights1 -= learning_rate * np.dot(training_data.T, gradient_hidden1)
    bias1 -= learning_rate * np.sum(gradient_hidden1, axis=0, keepdims=True)

# Evaluación del modelo
hidden1 = relu(np.dot(training_data, weights1) + bias1)
hidden2 = relu(np.dot(hidden1, weights2) + bias2)
hidden3 = relu(np.dot(hidden2, weights3) + bias3)
hidden4 = relu(np.dot(hidden3, weights4) + bias4)
hidden5 = relu(np.dot(hidden4, weights5) + bias5)
output = linear(np.dot(hidden5, weights6) + bias6)
scores = np.mean(np.abs(output - target_data))

print("Mean Absolute Error:", scores)
