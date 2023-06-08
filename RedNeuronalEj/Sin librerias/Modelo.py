import numpy as np

# Función de activación ReLU
def relu(x):
    """
    Función de activación ReLU (Rectified Linear Unit).
    Aplica una función de umbral a los elementos de entrada.
    Si el elemento es positivo, se mantiene el mismo valor.
    Si el elemento es negativo, se establece en cero.
    """
    return np.maximum(0, x)

# Capa de la red neuronal
class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        """
        Inicializa una capa de la red neuronal.

        Args:
        - input_size: Tamaño de entrada de la capa.
        - output_size: Tamaño de salida de la capa.
        - activation: Función de activación a aplicar en la capa.
        """
        self.weights = np.random.randn(input_size, output_size)  # Pesos de la capa
        self.biases = np.zeros((1, output_size))  # Sesgos de la capa
        self.activation = activation

    def forward(self, inputs):
        """
        Realiza la propagación hacia adelante en la capa.

        Args:
        - inputs: Entradas para la capa.

        Returns:
        - Salida de la capa después de aplicar la función de activación.
        """
        z = np.dot(inputs, self.weights) + self.biases  # Producto punto de las entradas con los pesos y sumar los sesgos
        a = self.activation(z)  # Aplicar la función de activación
        return a

# Definición de la red neuronal
class NeuralNetwork:
    def __init__(self):
        self.layers = []  # Lista de capas de la red neuronal

    def add_layer(self, layer):
        """
        Agrega una capa a la red neuronal.

        Args:
        - layer: Capa a agregar.
        """
        self.layers.append(layer)

    def forward(self, inputs):
        """
        Realiza la propagación hacia adelante en la red neuronal.

        Args:
        - inputs: Entradas para la red neuronal.

        Returns:
        - Salida de la red neuronal.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)  # Realizar la propagación hacia adelante en cada capa
        return inputs

# Creación de la red neuronal y las capas
network = NeuralNetwork()
network.add_layer(DenseLayer(1, 128, relu))
network.add_layer(DenseLayer(128, 68, relu))
network.add_layer(DenseLayer(68, 32, relu))
network.add_layer(DenseLayer(32, 16, relu))
network.add_layer(DenseLayer(16, 8, relu))
network.add_layer(DenseLayer(8, 1, lambda x: x))  # Capa de salida lineal

# Ejemplo de uso
input_data = np.array([[2.5]])  # Ejemplo de entrada
output = network.forward(input_data)  # Realizar la propagación hacia adelante en la red neuronal
print(output)
