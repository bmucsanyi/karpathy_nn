from __future__ import annotations

import math

from karpathy_nn.micrograd.utils import topological_sort


class Value:
    """Object that stores a single scalar value, its gradient wrt. another ``Value`` we
    call ``.backward()`` on, and the children of the node in the corresponding
    computational graph.

    """

    def __init__(
        self, data: float | int, label: str = "", _prev: tuple = (), _op: str = ""
    ) -> None:
        self.data = data
        # At initialization, we assume that the object does not impact
        # the variable we call .backward() on (the loss value L).
        self.grad = 0
        self.label = label
        # Internal variables used for autograd graph construction
        self._backward = lambda: None  # It will stay like this for a leaf node
        self._prev = set(_prev)
        self._op = _op

    def __repr__(self) -> str:
        # Default repr is very criptic, something like <__main__.Value at 0x7f9bb824b310>
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | float | int) -> Value:  # self + other
        # This is a convenience utility to be able to add python floats (or ints)
        # to Value objects, e.g. a + 2. For 2 + a, we have to define __radd__.
        other = other if isinstance(other, Value) else Value(data=other)

        # Note: the position of this function doesn't matter.
        # It will correctly refer to the out variable defined later.
        def _backward() -> None:
            # Gradients are accumulated. Refer to multivariate calculus.
            # If gradients are flowing from multiple out nodes into self
            # or other, not accumulating will lead to incorrect results.
            self.grad += out.grad
            other.grad += out.grad

        out = Value(data=self.data + other.data, _prev=(self, other), _op="+")
        out._backward = _backward

        return out

    def __radd__(self, other: Value | float | int) -> Value:  # other + self
        return self + other  # Commutativity

    def __sub__(self, other: Value | float | int) -> Value:  # self - other
        return self + (-other)

    def __rsub__(self, other: Value | float | int) -> Value:  # other - self
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __neg__(self) -> Value:  # -self
        return self * -1

    def __mul__(self, other: Value | float | int) -> Value:  # self * other
        other = other if isinstance(other, Value) else Value(data=other)

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out = Value(data=self.data * other.data, _prev=(self, other), _op="*")
        out._backward = _backward

        return out

    def __rmul__(self, other: Value | float | int) -> Value:
        return self * other  # Commutativity

    def __truediv__(self, other: Value | float | int) -> Value:  # self / other
        other = other if isinstance(other, Value) else Value(data=other)

        return self * other**-1

    def __rtruediv__(self, other: Value | float | int) -> Value:  # other / self
        return other * self**-1

    def __pow__(self, other: float | int) -> Value:  # self ** other
        other = other if isinstance(other, Value) else Value(data=other)

        def _backward() -> None:
            if self.data == 0 and other.data < 1:
                self.grad += float("inf")  # This is how PyTorch handles this
            else:
                self.grad += other.data * self.data ** (other.data - 1) * out.grad

            if self.data > 0:
                other.grad += self.data**other.data * math.log(self.data) * out.grad
            elif self.data == 0:
                other.grad += 0
            else:
                other.grad += float("nan")

        if self.data == 0 and other.data < 0:
            data = float("inf")
        else:
            data = self.data**other.data

        out = Value(data=data, _prev=(self, other), _op=f"**")
        out._backward = _backward

        return out

    def __rpow__(self, other: Value | float) -> Value:  # other ** self
        other = other if isinstance(other, Value) else Value(data=other)

        return other**self

    def exp(self) -> Value:
        """Implements the exp function.

        Returns:
            A ``Value`` object whose ``data`` field is ``self.data`` transformed by the
            exp function.

        """
        # We could define new autograd functions in PyTorch as shown in
        # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

        def _backward() -> None:
            self.grad += out.data * out.grad

        out = Value(data=math.exp(self.data), _prev=(self,), _op="exp")
        out._backward = _backward

        return out

    def log(self) -> Value:
        """Implements the (natural) log function.

        Returns:
            A ``Value`` object whose ``data`` field is ``self.data`` transformed by the
            log function.

        """

        def _backward() -> None:
            self.grad += 1 / self.data * out.grad

        if self.data < 0:
            data = float("nan")
        elif self.data == 0:
            data = -float("inf")
        else:
            data = math.log(self.data)

        out = Value(data=data, _prev=(self,), _op="log")
        out._backward = _backward

        return out

    def tanh(self) -> Value:
        """Implements the tanh function.

        Returns:
            A ``Value`` object whose ``data`` field is ``self.data`` transformed by the
            tanh function.

        """

        # We can create functions at arbitrary points of abstraction.
        # What changes is how hard it is to implement the backward pass
        # through the function (i.e., the local derivative).
        # We can directly implement tanh without needing to implement exp first,
        # as long as we provide correct local derivatives.
        def _backward() -> None:
            # Again, there might be multiple out values that self affects.
            # We have to sum the individual gradients.
            # PyTorch version:
            # https://github.com/pytorch/pytorch/blob/c5872e6d6d8fd9b8439b914c143d49488335f573/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L793-L836
            self.grad += (1 - out.data**2) * out.grad

        x = self.data
        tanh_data = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(data=tanh_data, _prev=(self,), _op="tanh")
        out._backward = _backward

        return out

    def relu(self) -> Value:
        """Implements the relu function.

        Returns:
            A ``Value`` object whose ``data`` field is ``self.data`` transformed by the
            relu function.

        """

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out = Value(data=0 if self.data < 0 else self.data, _prev=(self,), _op="ReLU")
        out._backward = _backward

        return out

    def sigmoid(self) -> Value:
        """Implements the sigmoid function.

        Returns:
            A ``Value`` object whose ``data`` field is ``self.data`` transformed by the
            sigmoid function.

        """

        def _backward() -> None:
            self.grad += out.data * (1 - out.data) * out.grad

        data = 1 / (1 + math.exp(-self.data))
        out = Value(data=data, _prev=(self,), _op="sigmoid")
        out._backward = _backward

        return out

    def backward(self) -> None:
        """Implements backpropagation in the computational graph of the variable.

        Notably, all nodes' gradients are populated that are in the computational graph.

        """
        # To call the _backward functions in the right order, we need to first
        # obtain the topological ordering of the computational graph
        # which is a layout of nodes such that all edges "go only one way",
        # from left to right.
        self.grad = 1
        topological_ordering = topological_sort(self)

        for node in reversed(topological_ordering):
            node._backward()
