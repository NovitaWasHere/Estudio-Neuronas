import numpy as np

# Datos de entrenamiento
# Entradas
X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7.2, 8.4, 9, 10, 11, 12, 13, 14, 15, 16, 17.2, 18.4, 19, 20])
# Salidas esperadas
y_train = np.array([0, 2, 4, 6, 8, 10, 12, 14.4, 16.8, 18, 20, 22, 24, 26, 28, 30, 32, 34.4, 36.8, 38, 40])

# Inicialización de pesos y bias
w = np.random.rand()  # Peso
b = np.random.rand()  # Bias

# Hiperparámetros
learning_rate = 0.001
epochs = 2000

# Ciclo de entrenamiento
for epoch in range(epochs):
    # Forward pass
    y_pred = X_train * w + b

    # Cálculo del error
    error = y_pred - y_train

    # Cálculo de los gradientes
    w_gradient = np.mean(error * X_train)
    b_gradient = np.mean(error)

    # Actualización de pesos y bias
    w -= learning_rate * w_gradient
    print(w)
    b -= learning_rate * b_gradient
    print(b)

# Prueba con nuevos datos
numeroDado = float(input("Numero"))
X_test = np.array([numeroDado, 8, 7234])
y_pred = X_test * w + b
print(y_pred)
