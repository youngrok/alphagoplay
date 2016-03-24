import random

import math


class Neuron:

    def __init__(self, layer, number):
        self.value = 0
        self.layer = layer
        self.number = number

    def __str__(self):
        return '%s %s: %s' % (self.layer, self.number, self.value)


class Synapse:
    all = []
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.weight = random.random()
        self.all.append(self)

    def __str__(self):
        return '%s -- %s --> %s' % (self.a, self.b, self.weight)

    @classmethod
    def print(cls):
        for s in cls.all:
            print(s)


class NeuralNetwork:

    def __init__(self):
        self.input = [Neuron('input', 0), Neuron('input', 1)]
        self.hidden = [Neuron('hidden', 0), Neuron('hidden', 1), Neuron('hidden', 2)]
        self.output = [Neuron('output', 0)]

        for i in self.input:
            i.synapses = [Synapse(i, h) for h in self.hidden]

        for h in self.hidden:
            h.synapses = [Synapse(h, self.output[0])]

    def forward(self, input):
        self.input[0].value = input[0]
        self.input[1].value = input[1]

        for h in self.hidden:
            h.value = 0

        for o in self.output:
            o.value = 0

        for i in self.input:
            for s in i.synapses:
                s.b.value += i.value * s.weight

        for h in self.hidden:
            for s in h.synapses:
                s.b.value += sigmoid(h.value) * s.weight

        return sigmoid(self.output[0].value)

    def back(self, expected, output):
        delta_output = expected - output
        delta_sum = delta_output / dsigmoid(self.output[0].value)

        for h in self.hidden:
            for s in h.synapses:
                s.weight += (delta_sum / h.value)

    def expected_sum(self, sum, expected, output):
        delta_output = expected - output
        return sum + delta_output / dsigmoid(sum)

    def train(self, input, expected):
        output = self.forward(input)
        self.back(expected, output)


def sigmoid(x):
    if x < -700: x = -700
    return 1 / (1 + math.exp(-x))


def dsigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)
    # if x < -350: x = -350
    # return -math.exp(-x) / (1 + 2 * math.exp(-x) + math.exp(-2 * x))


ann = NeuralNetwork()
print('result:', ann.forward([1, 1]))
ann.train([1, 1], 0)
print('result:', ann.forward([1, 1]))

