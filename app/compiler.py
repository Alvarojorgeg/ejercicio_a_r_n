"""Interpreter that translates a textual architecture into a simple Sequential model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class LayerSpec:
    layer_type: str
    units: int
    activation: str


class DenseLayer:
    def __init__(self, units: int, activation: str):
        self.units = units
        self.activation = activation

    def summary_line(self, input_dim: int) -> str:
        params = input_dim * self.units + self.units
        return f"dense (Dense)         ({input_dim}, {self.units})       {params:>6}    activation={self.activation}"


class SequentialModel:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.layers: List[DenseLayer] = []

    def add(self, layer: DenseLayer) -> None:
        self.layers.append(layer)

    def summary(self) -> str:
        lines = [
            "Model: \"sequential\"",
            "Layer (type)           Output Shape        Param #",
            "==================================================",
        ]
        current_dim = self.input_dim
        total_params = 0
        for layer in self.layers:
            lines.append(layer.summary_line(current_dim))
            total_params += current_dim * layer.units + layer.units
            current_dim = layer.units
        lines.append("==================================================")
        lines.append(f"Total params: {total_params}")
        return "\n".join(lines)


SUPPORTED_LAYERS: dict[str, Callable[[LayerSpec], DenseLayer]] = {
    "dense": lambda spec: DenseLayer(units=spec.units, activation=spec.activation),
}


def parse_layer(layer_str: str) -> LayerSpec:
    name, _, args = layer_str.partition("(")
    if not _ or not args.endswith(")"):
        raise ValueError(f"Layer definition '{layer_str}' is malformed")
    args = args[:-1]
    parts = [part.strip() for part in args.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Layer definition must provide number of units and activation, e.g. Dense(128, relu)")
    units = int(parts[0])
    activation = parts[1]
    return LayerSpec(layer_type=name.strip().lower(), units=units, activation=activation)


def compile_model(architecture: str, input_dim: int = 784) -> SequentialModel:
    model = SequentialModel(input_dim=input_dim)
    layers = [segment.strip() for segment in architecture.split("->") if segment.strip()]
    if not layers:
        raise ValueError("Architecture string must contain at least one layer definition")
    for layer_str in layers:
        spec = parse_layer(layer_str)
        factory = SUPPORTED_LAYERS.get(spec.layer_type)
        if factory is None:
            raise ValueError(f"Unsupported layer type: {spec.layer_type}")
        model.add(factory(spec))
    return model


__all__ = ["compile_model", "parse_layer", "LayerSpec", "SequentialModel", "DenseLayer"]
