"""Training utilities for the custom Sequential model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .compiler import SequentialModel, compile_model


@dataclass
class PredictionExample:
    pixels: List[int]
    label: int
    prediction: int
    correct: bool


@dataclass
class TrainingArtifacts:
    model: SequentialModel
    weights: List[List[List[float]]]
    biases: List[List[float]]
    history: List[Tuple[float, float]]
    evaluation: Tuple[float, float]
    architecture: str
    training_log: str
    summary: str
    examples: List[PredictionExample]
    epochs: int
    learning_rate: float
    noise: float


DEFAULT_ARCHITECTURE = "Dense(16, relu) -> Dense(16, relu) -> Dense(10, softmax)"
INPUT_DIM = 196


_DIGIT_PATTERNS = {
    0: [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    1: [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    2: [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    3: [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    4: [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    5: [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
    6: [
        "01110",
        "10000",
        "11110",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    7: [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    8: [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    9: [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100",
    ],
}


def _scale_pattern(pattern: List[str], scale: int = 2, target_size: int = 14) -> List[float]:
    height = len(pattern)
    width = len(pattern[0])
    grid = [[0 for _ in range(width * scale)] for _ in range(height * scale)]
    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            value = 255 if ch == "1" else 0
            for dr in range(scale):
                for dc in range(scale):
                    grid[r * scale + dr][c * scale + dc] = value
    padded = [[0 for _ in range(target_size)] for _ in range(target_size)]
    row_offset = (target_size - height * scale) // 2
    col_offset = (target_size - width * scale) // 2
    for r in range(height * scale):
        for c in range(width * scale):
            padded[row_offset + r][col_offset + c] = grid[r][c]
    flat = [pixel / 255.0 for row in padded for pixel in row]
    return flat


def _build_dataset(noise: float = 0.0) -> Tuple[List[List[float]], List[int]]:
    samples: List[List[float]] = []
    labels: List[int] = []
    for digit, pattern in _DIGIT_PATTERNS.items():
        vector = _scale_pattern(pattern)
        for _ in range(20):
            if noise > 0:
                jittered = []
                for value in vector:
                    jitter = random.uniform(-noise, noise)
                    jittered.append(min(1.0, max(0.0, value + jitter)))
                samples.append(jittered)
            else:
                samples.append(list(vector))
            labels.append(digit)
    return samples, labels


def _one_hot(label: int, num_classes: int = 10) -> List[float]:
    vec = [0.0] * num_classes
    vec[label] = 1.0
    return vec


def _activation_fn(name: str):
    name = name.lower()
    if name == "relu":
        return lambda vec: [max(0.0, v) for v in vec]
    if name == "sigmoid":
        return lambda vec: [1.0 / (1.0 + math.exp(-v)) for v in vec]
    if name == "softmax":
        def softmax(vec: Sequence[float]) -> List[float]:
            m = max(vec)
            exps = [math.exp(v - m) for v in vec]
            total = sum(exps)
            return [v / total for v in exps]
        return softmax
    return lambda vec: list(vec)


def _activation_derivative(name: str, activated: Sequence[float], pre_activation: Sequence[float]) -> List[float]:
    name = name.lower()
    if name == "relu":
        return [1.0 if z > 0 else 0.0 for z in pre_activation]
    if name == "sigmoid":
        return [a * (1.0 - a) for a in activated]
    if name == "softmax":
        # Softmax derivative handled separately with cross-entropy
        return [1.0] * len(activated)
    return [1.0] * len(activated)


def _initialize_parameters(model: SequentialModel) -> Tuple[List[List[List[float]]], List[List[float]]]:
    random.seed(42)
    dims = [model.input_dim] + [layer.units for layer in model.layers]
    weights: List[List[List[float]]] = []
    biases: List[List[float]] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        layer_weights = [[random.uniform(-limit, limit) for _ in range(in_dim)] for _ in range(out_dim)]
        layer_biases = [0.0 for _ in range(out_dim)]
        weights.append(layer_weights)
        biases.append(layer_biases)
    return weights, biases


def _forward(model: SequentialModel, weights, biases, inputs: List[float]):
    activations = [inputs]
    pre_activations = []
    current = inputs
    for layer, layer_weights, layer_biases in zip(model.layers, weights, biases):
        z = []
        for neuron_weights, bias in zip(layer_weights, layer_biases):
            z.append(sum(w * x for w, x in zip(neuron_weights, current)) + bias)
        pre_activations.append(z)
        activation = _activation_fn(layer.activation)
        current = activation(z)
        activations.append(current)
    return activations, pre_activations


def _backprop(model: SequentialModel, weights, biases, activations, pre_activations, target: List[float]):
    grads_w = [[[0.0 for _ in neuron] for neuron in layer] for layer in weights]
    grads_b = [[0.0 for _ in layer] for layer in biases]

    # Output layer gradient with softmax + cross-entropy
    last_layer = model.layers[-1]
    output_activation = activations[-1]
    delta = [output_activation[i] - target[i] for i in range(len(target))]
    grads_b[-1] = delta
    for i, neuron_delta in enumerate(delta):
        for j, activation_value in enumerate(activations[-2]):
            grads_w[-1][i][j] = neuron_delta * activation_value

    # Hidden layers
    for layer_index in range(len(model.layers) - 2, -1, -1):
        layer = model.layers[layer_index]
        derivative = _activation_derivative(layer.activation, activations[layer_index + 1], pre_activations[layer_index])
        new_delta = []
        for neuron_index in range(len(weights[layer_index])):
            error = 0.0
            for k, next_weights in enumerate(weights[layer_index + 1]):
                error += next_weights[neuron_index] * delta[k]
            new_delta.append(error * derivative[neuron_index])
        delta = new_delta
        grads_b[layer_index] = delta
        for i, neuron_delta in enumerate(delta):
            for j, activation_value in enumerate(activations[layer_index]):
                grads_w[layer_index][i][j] = neuron_delta * activation_value

    return grads_w, grads_b


def _update_parameters(weights, biases, grads_w, grads_b, learning_rate):
    for layer_idx in range(len(weights)):
        for neuron_idx in range(len(weights[layer_idx])):
            biases[layer_idx][neuron_idx] -= learning_rate * grads_b[layer_idx][neuron_idx]
            for weight_idx in range(len(weights[layer_idx][neuron_idx])):
                weights[layer_idx][neuron_idx][weight_idx] -= learning_rate * grads_w[layer_idx][neuron_idx][weight_idx]


def _train(
    model: SequentialModel,
    samples: List[List[float]],
    labels: List[int],
    epochs: int = 30,
    learning_rate: float = 0.05,
):
    weights, biases = _initialize_parameters(model)
    history: List[Tuple[float, float]] = []
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        combined = list(zip(samples, labels))
        random.shuffle(combined)
        for sample, label in combined:
            target = _one_hot(label)
            activations, pre_activations = _forward(model, weights, biases, sample)
            probs = activations[-1]
            loss = -math.log(max(probs[label], 1e-9))
            total_loss += loss
            if max(range(len(probs)), key=lambda i: probs[i]) == label:
                correct += 1
            grads_w, grads_b = _backprop(model, weights, biases, activations, pre_activations, target)
            _update_parameters(weights, biases, grads_w, grads_b, learning_rate)
        avg_loss = total_loss / len(samples)
        accuracy = correct / len(samples)
        history.append((avg_loss, accuracy))
    return weights, biases, history


def _predict(model: SequentialModel, weights, biases, sample: List[float]) -> List[float]:
    activations, _ = _forward(model, weights, biases, sample)
    return activations[-1]


def _format_history(history: List[Tuple[float, float]]) -> str:
    lines = []
    for idx, (loss, acc) in enumerate(history, start=1):
        lines.append(f"Epoch {idx:02d} - loss: {loss:.4f} - acc: {acc:.4f}")
    return "\n".join(lines)


def train_mnist(
    architecture: str = DEFAULT_ARCHITECTURE,
    epochs: int = 20,
    learning_rate: float = 0.05,
    noise: float = 0.0,
) -> TrainingArtifacts:
    model = compile_model(architecture, input_dim=INPUT_DIM)
    samples, labels = _build_dataset(noise=noise)
    weights, biases, history = _train(
        model,
        samples,
        labels,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    loss, acc = history[-1]

    # Build prediction examples using one instance per digit
    examples = []
    for digit, pattern in _DIGIT_PATTERNS.items():
        vector = _scale_pattern(pattern)
        probs = _predict(model, weights, biases, vector)
        prediction = max(range(len(probs)), key=lambda i: probs[i])
        pixels = [int(v * 255) for v in vector]
        examples.append(
            PredictionExample(
                pixels=pixels,
                label=digit,
                prediction=prediction,
                correct=prediction == digit,
            )
        )

    summary = model.summary()
    training_log = "\n".join(
        [
            f"Configuración: epochs={epochs}, learning_rate={learning_rate:.4f}, noise={noise:.2f}",
            _format_history(history),
        ]
    )
    evaluation = (loss, acc)

    return TrainingArtifacts(
        model=model,
        weights=weights,
        biases=biases,
        history=history,
        evaluation=evaluation,
        architecture=architecture,
        training_log=training_log,
        summary=summary,
        examples=examples,
        epochs=epochs,
        learning_rate=learning_rate,
        noise=noise,
    )


__all__ = ["train_mnist", "TrainingArtifacts", "PredictionExample", "DEFAULT_ARCHITECTURE"]
