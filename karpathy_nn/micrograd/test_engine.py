import torch

from karpathy_nn.micrograd.engine import Value

# Run: execute
# $ pytest
# in the micrograd folder.


def test_sanity_check():
    x = Value(data=-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_micrograd, y_micrograd = x, y

    x = torch.tensor(-4.0, dtype=torch.double, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_torch, y_torch = x, y

    # Forward pass went well:
    assert y_micrograd.data == y_torch.data.item()
    # Backward pass went well:
    assert x_micrograd.grad == x_torch.grad.item()


def test_more_ops():
    a = Value(data=-4.0)
    b = Value(data=2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10 / f
    g.backward()
    a_micrograd, b_micrograd, g_micrograd = a, b, g

    a = torch.tensor(-4.0, dtype=torch.double, requires_grad=True)
    b = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10 / f
    g.backward()
    a_torch, b_torch, g_torch = a, b, g

    tol = 1e-6
    # Forward pass went well:
    assert abs(g_micrograd.data - g_torch.data.item()) < tol
    # Backward pass went well:
    assert abs(a_micrograd.grad - a_torch.grad.item()) < tol
    assert abs(b_micrograd.grad - b_torch.grad.item()) < tol
