import numpy as np

"""
    Для предыдущей задачи реализовать обучение нейронной сети с двумя нейронами по правилу Хебба.
"""
# Формулировка правила Хебба звучит следующим образом:
# "Когда аксон нейрона A возбуждает нейрон B и продолжает это делать, связь между этими нейронами усиливается"

# Так как мало слоев и мало обучающих примеров выводит большие значения нейронов и не использовали shuffle

# Класс нейронной сети
class Hebb_Network:
    def __init__(self, input_size, num_neurons):
        # Рандомим веса
        self.weights = np.random.rand(num_neurons, input_size)

    def train(self, inputs, num_iterations):
        for i in range(num_iterations):
            for x in inputs:
                output = np.dot(self.weights, x)
                for j in range(len(self.weights)):
                    if output[j] > 0:
                        self.weights[j] += x
                    else:
                        self.weights[j] -= x

    def predict(self, inputs):
        activations = np.dot(self.weights, inputs)
        winner_index = np.argmax(activations)
        return winner_index


# Входные обучающие векторы
training_inputs = np.array([
    # нормализованные векторы
    [0.97, 0.2],
    [1.0, 0.0],
    [-0.72, 0.7],
    [-0.67, 0.74],
    [-0.8, 0.6],
    [0.0, -1.0],
    [0.20, -0.97],
    [-0.3, -0.95],
])

# Нейронная сеть с 2 нейронами
network = Hebb_Network(input_size=2, num_neurons=4)
print("До обучения:")
print(network.weights)
print("")
# Обучение сети по входным обучающим векторам
network.train(training_inputs, num_iterations=120)

# Вывод выходных весов нейронов
print("После обучения:")
print(network.weights)
