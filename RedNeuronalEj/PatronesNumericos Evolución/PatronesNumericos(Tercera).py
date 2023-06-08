import numpy as np


# Función de activación (ReLU)
def relu(x):
    return np.maximum(0, x)


# Datos de entrenamiento
X_train = np.array([[2], [4], [6], [8], [10]])  # Entradas
y_train = np.array([[4], [8], [12], [16], [20]])  # Salidas esperadas

# Inicialización de pesos y bias
np.random.seed(0)
W1 = np.random.rand(1, 4)  # Pesos capa oculta
b1 = np.random.rand(4)  # Bias capa oculta
W2 = np.random.rand(4, 1)  # Pesos capa de salida
b2 = np.random.rand()  # Bias capa de salida

# Hiperparámetros
learning_rate = 0.01
epochs = 1000

# Ciclo de entrenamiento
for epoch in range(epochs):
    # Forward pass
    hidden_layer = relu(np.dot(X_train, W1) + b1)
    y_pred = np.dot(hidden_layer, W2) + b2

    # Cálculo del error
    error = y_pred - y_train

    # Cálculo de los gradientes y actualización de pesos y bias
    dW2 = np.dot(hidden_layer.T, error)
    db2 = np.sum(error)
    dhidden = np.dot(error, W2.T) * (hidden_layer > 0)
    dW1 = np.dot(X_train.T, dhidden)
    db1 = np.sum(dhidden, axis=0)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# Obtener entrada del usuario
input_num = float(input("Ingrese un número: "))

# Preparar dato de prueba
X_test = np.array([[input_num]])

# Realizar forward pass con los pesos y bias entrenados
hidden_layer = relu(np.dot(X_test, W1) + b1)
y_pred_test = np.dot(hidden_layer, W2) + b2

# Imprimir resultado
print("El número duplicado es:", y_pred_test[0][0])

