"""Pure Python implementation of the forward pass of a simple MLP.

Although the original activity references NumPy, the execution environment does
not provide third-party packages. This module therefore implements the required
behaviour using plain Python lists while keeping the same conceptual structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


Activation = Callable[[List[float]], List[float]]
Vector = List[float]
Matrix = List[List[float]]


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid_scalar(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def relu_scalar(x: float) -> float:
    return x if x > 0.0 else 0.0


def _apply_activation(values: Vector, fn: Callable[[float], float]) -> Vector:
    return [fn(v) for v in values]


def sigmoid(values: Vector) -> Vector:
    return _apply_activation(values, sigmoid_scalar)


def relu(values: Vector) -> Vector:
    return _apply_activation(values, relu_scalar)


@dataclass
class Neuron:
    weights: Vector
    bias: float
    activation: Callable[[Vector], Vector]

    def forward(self, inputs: Vector) -> float:
        z = _dot(inputs, self.weights) + self.bias
        return self.activation([z])[0]


class Layer:
    def __init__(self, weights: Matrix, biases: Vector, activation: Callable[[Vector], Vector]):
        if len(weights) == 0:
            raise ValueError("Weight matrix must not be empty")
        if len(weights[0]) != len(biases):
            raise ValueError("Bias vector length must match the number of neurons")
        self.neurons: List[Neuron] = [
            Neuron(weights=[row[i] for row in weights], bias=biases[i], activation=lambda v, fn=activation: fn(v))
            for i in range(len(biases))
        ]

    def forward(self, inputs: Vector) -> Vector:
        return [neuron.forward(inputs) for neuron in self.neurons]


class MLP:
    def __init__(self, layers: Sequence[Layer]):
        if not layers:
            raise ValueError("An MLP requires at least one layer")
        self.layers = list(layers)

    def predict(self, inputs: Vector) -> Vector:
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations


__all__ = ["sigmoid", "relu", "Neuron", "Layer", "MLP"]
