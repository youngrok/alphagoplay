import random

import math


class Neuron:

    def __init__(self):
        self.value = 0


class Synapse:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.weight = random.random()


class NeuralNetwork:

    def __init__(self):
        self.input = [Neuron(), Neuron()]
        self.hidden = [Neuron(), Neuron(), Neuron()]
        self.output = [Neuron()]

        for i in self.input:
            i.synapses = [Synapse(i, h) for h in self.hidden]

        for h in self.hidden:
            h.synapses = [Synapse(i, self.output[0])]

    def forward(self, *input):
        self.input[0].value = input[0]
        self.input[1].value = input[1]

        for i in self.input:
            for s in i.synapses:
                s.b.value += i.value * s.weight

        for h in self.hidden:
            h.value = sigmoid(h.value)
            for s in h.synapses:
                s.b.value += h.value * s.weight

        return sigmoid(self.output[0].value)


def sigmoid(x):
    return 1 / (1 - math.exp(-x))


ann = NeuralNetwork()
print(ann.forward(1, 1))

