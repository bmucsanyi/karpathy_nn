import random

from karpathy_nn.micrograd.engine import Value


class Module:
    """Base module class."""

    def zero_grad(self):
        """Zeros out every parameter's gradients in the module."""
        for parameter in self.parameters():
            parameter.grad = 0

    def parameters(self):
        """Returns the parameters of the module in a list."""
        return []


class Neuron(Module):
    """Basic neuron class."""

    def __init__(self, num_input_neurons: int, nonlinearity: str = "Tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_input_neurons)]
        self.b = Value(data=0)  # Controls overall trigger-happiness of the neuron
        self.nonlinearity = nonlinearity

    def __call__(self, x: list[float | int]) -> Value:
        pre_activation = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)

        if self.nonlinearity == "ReLU":
            return pre_activation.relu()

        if self.nonlinearity == "Tanh":
            return pre_activation.tanh()

        if self.nonlinearity == "Sigmoid":
            return pre_activation.sigmoid()

        if self.nonlinearity == "None":
            return pre_activation

    def parameters(self) -> list[Value]:
        """Returns all parameters of the neuron in a list."""
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{self.nonlinearity}Neuron({len(self.w)})"


class Layer(Module):
    """An aggregation of ``Neuron``s into a fully-connected layer."""

    def __init__(
        self,
        num_input_neurons: int,
        num_output_neurons: int,
        nonlinearity: str = "ReLU",
    ) -> None:
        self.neurons = [
            Neuron(num_input_neurons, nonlinearity) for _ in range(num_output_neurons)
        ]

    def __call__(self, x: list[float | int]) -> Value | list[Value]:
        # A layer of neurons is just a set of neurons evaluated independently.
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        """Returns all parameters of the layer in a list."""
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"


class MLP(Module):
    """An aggregation of fully-connected ``Layer``s into a Multi-Layer Perceptron."""

    def __init__(
        self,
        num_input_neurons: int,
        num_output_neurons_per_layer: list[int],
        final_act: str = "Sigmoid",
    ) -> None:
        all_nums = [num_input_neurons] + num_output_neurons_per_layer
        num_layers = len(num_output_neurons_per_layer)
        self.layers = [
            Layer(
                all_nums[i],
                all_nums[i + 1],
                nonlinearity=final_act if i == num_layers - 1 else "ReLU",
            )
            # Layer(all_nums[i], all_nums[i + 1], nonlinearity="Tanh")
            for i in range(num_layers)
        ]

    def __call__(self, x: list[float | int]) -> Value | list[Value]:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> list[Value]:
        """Returns all parameters of the MLP in a list."""
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
