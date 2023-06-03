import math
import numpy as np
import pandas as pd


class Percept:

    # Статическая переменная, хранящая скорость обучения
    LEARNING_RATE = 0.3

    # Конструктор класса, инициализирует входные данные (X), количество признаков, и создает список из четырех нейронов.
    def __init__(self, X, amount_of_features, amount_of_hidden=4):
        self.__X = X
        self.__neurons = [Percept.Neuron(num_of_neuron=i, amount_of_features=amount_of_features) for i in range(amount_of_hidden)]
        self.clusters = None

    # Функция для обучения сети на заданное количество эпох (по умолчанию 6). В течение обучения, скорость обучения
    # уменьшается на 0.05 после каждой эпохи.
    def transform_fit(self, epochs=6):
        self.clusters = []
        Percept.LEARNING_RATE = 0.3
        samples = [x for x in self.__X.values]
        for epoch in range(epochs):
            np.random.permutation(samples)
            self.__influence_of_signal(samples, epoch=epoch)
            Percept.LEARNING_RATE -= 0.05

        self.__influence_of_signal([x for x in self.__X.values], is_predict=True)  # clustering samples

    # Вспомогательная функция для обработки сигналов и корректировки весов в зависимости от эпохи или для кластеризации данных.
    def __influence_of_signal(self, samples, is_predict=False, epoch=0):
        for sample in samples:
            rs = []  # distances
            for neuron in self.__neurons:
                rs.append(neuron.find_metrics(sample))
            idx = pd.Series(rs).idxmin()
            if not is_predict:
                self.__neurons[idx].correlate_weights(sample, epoch=epoch)
            else:
                # print(rs)
                self.clusters.append(idx)

    class Neuron:

        # Конструктор класса, инициализирует номер нейрона, его веса и расстояние.
        def __init__(self, num_of_neuron, amount_of_features):
            self.__num_of_neuron = num_of_neuron
            self.__ws = [np.random.uniform(-1, 1) for i in range(amount_of_features)]
            self.__distance = None

        # Функция для вычисления евклидова расстояния между входными данными и весами нейрона.
        def find_metrics(self, sample):  # Euclid`s distance
            distance = 0
            for x, w in zip(sample, self.__ws):
                distance += math.pow(x - w, 2)
            self.__distance = math.sqrt(distance)
            return self.__distance

        # Функция для корректировки весов нейрона с использованием скорости обучения.
        def correlate_weights(self, X, epoch):
            for i in range(len(self.__ws)):
                self.__ws[i] = self.__ws[i] + Percept.LEARNING_RATE * (X[i] - self.__ws[i])

        #  Функция для вычисления функции близости, которая не используется в данной реализации
        def proximity_function(self, epoch):
            sigma = 1 / math.exp(math.pow(epoch+1, -2))
            return math.exp(-1/sigma)
            # pass


# X = pd.DataFrame({
#     0: [0,0,1,0,1,1,1,0,0,0,1,1,0,1,0,1,0,1,0,0],
#     1: [1,0,0,1,1,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0],
#     2: [60,60,60,85,65,60,55,55,55,60,85,60,55,80,55,60,75,85,80,55],
#     3: [79,61,61,78,78,78,79,56,60,56,89,88,64,83,10,67,98,85,56,60],
#     4: [60,30,30,72,60,77,56,50,21,30,85,76,0,62,3,57,86,81,50,30],
#     5: [72,5,66,70,67,81,69,56,64,16,92,66,9,72,8,64,82,85,69,8],
#     6: [63,17,58,85,65,60,72,60,50,17,85,60,50,72,50,50,85,72,50,60],
# })
#
# for i in range(len(X.columns)-2):
#     X[i+2] = X[i+2].apply(lambda x: (x - X[i+2].values.min()) / (X[i+2].values.max() - X[i+2].values.min()))
# p = Percept(X, amount_of_features=len(X.columns))
# p.transform_fit()
# print(p.clusters)

# 24 мая - 14:10
# Подколзин Миков Лукащик
# Уварова Полупанов Гаркуша
# Синица Добровольская Полетайкин