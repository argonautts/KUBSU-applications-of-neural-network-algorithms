import numpy as np

"""
    Для предыдущей задачи использовать модифицированное обучение WTA (например, ввести систему штрафов: 
    учитывать прошлые победы каждого нейрона и штрафовать те нейроны, которые побеждали больше всего. 
    Штрафование может назначаться либо при достижении порогового значения числа побед, 
    либо уменьшением значения функции активации при нарастании количества побед).
"""

# Класс нейронов WTA
class WTANeuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.victories = 0
        self.penalty_factor = 1.0

    def activate(self, inputs):
        activation = np.dot(self.weights, inputs)
        return activation * self.penalty_factor

    def update_weights(self, inputs, learning_rate):
        winner_index = np.argmax(inputs)
        if winner_index == np.argmax(self.weights):
            self.victories += 1
            if self.victories >= 3:
                self.penalty_factor *= 0.5
                winner_index = 0
        else:
            self.victories = 0
            self.penalty_factor = 1.0
        self.weights[winner_index] += learning_rate * (inputs[winner_index] - self.weights[winner_index])


# Класс нейронной сети
class WTA_Network:
    def __init__(self, input_size, num_neurons):
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(WTANeuron(input_size))

    def train(self, inputs, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(inputs, learning_rate)

    def predict(self, inputs):
        activations = []
        for neuron in self.neurons:
            activations.append(neuron.activate(inputs))
        winner_index = np.argmax(activations)
        return winner_index


# Входные обучающие векторы
training_inputs = [
    # нормализованные векторы
    [0.97, 0.2],
    [1.0, 0.0],
    [-0.72, 0.7],
    [-0.67, 0.74],
    [-0.8, 0.6],
    [0.0, -1.0],
    [0.20, -0.97],
    [-0.3, -0.95],
]

# Нейронная сеть WTA с 4 нейронами
wta_network = WTA_Network(input_size=2, num_neurons=4)

print("До обучения:")
for i, neuron in enumerate(wta_network.neurons):
    print("Neuron {}: {}".format(i, neuron.weights))

# Обучение сети по входным обучающим векторам
learning_rate = 0.5
for inputs in training_inputs:
    wta_network.train(inputs, learning_rate)


# Вывод выходных весов нейронов
print("\nПосле обучения:")
for i, neuron in enumerate(wta_network.neurons):
    print("Neuron {}: {}".format(i, neuron.weights))
