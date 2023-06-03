import numpy as np

"""
    Реализовать нейронную сеть на языке программирования Python, состоящую из четырех нейронов типа WTA, 
    предназначенную для классификации входных двухкомпонентных векторов.
    В качестве входных обучающих векторов использовать векторы в нормализованном виде.
    Вывести веса нейронов после обучения для коэффициента обучения n = 0.5
"""

# Класс нейронов WTA
class WTANeuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)

    def activate(self, inputs):
        # Вычисляет скалярное произведение вектора весов и вектора входных значений.
        # Это дает линейную комбинацию входных значений, где каждое значение умножено на соответствующий вес
        activation = np.dot(self.weights, inputs)
        return activation

    def update_weights(self, inputs, learning_rate):
        # Находит индекс нейрона, который имеет наибольшее значение входных данных.
        # Этот нейрон называется победителем (winner neuron), потому что он более активен, чем остальные нейроны в сети.
        winner_index = np.argmax(inputs)
        # Обновляет веса победившего нейрона и его ближайших соседей. Это делается путем изменения весов на основе разницы
        # между входными данными и текущими значениями весов.
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
            # Метод вычисляет активацию нейрона и добавляет ее в список activations
            activations.append(neuron.activate(inputs))
        # Находит индекс нейрона с максимальной активацией, используя функцию argmax, и возвращает этот индекс в качестве предсказания.
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

#  Нейронная сеть WTA с 4 нейронами
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
