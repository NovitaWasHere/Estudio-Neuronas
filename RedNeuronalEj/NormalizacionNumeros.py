import numpy as np

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
