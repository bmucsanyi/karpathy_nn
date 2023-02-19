import random
from typing import TypeVar

import numpy as np
import torch
from graphviz import Digraph

Node = TypeVar("Node")
Edge = tuple[Node, Node]


def trace(root: Node) -> tuple[set[Node], set[Edge]]:
    """Builds a set of all nodes and edges in a (forward) computational graph.

    Args:
        root: The root of the computational graph.

    Returns:
        All nodes and edges in the computational graph defined by ``root``.

    """
    nodes, edges = set(), set()

    def build(node: Node) -> None:
        """A recursive function that populates ``nodes`` and ``edges``."""
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.add((child, node))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(root: Node) -> Digraph:
    """Creates a ``Digraph`` representation of the (forward) computational graph corresponding to ``root``.

    Args:
        root: The variable whose computation graph is to be visualized.

    Returns:
        A ``Digraph`` object representing the computational graph of ``root``.

    """
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for node in nodes:
        node_id = str(id(node))

        # For any value in the graph, create a rectangular ("record") node.
        graph.node(
            name=node_id,
            label=f"{{{node.label} | data {node.data:.4f} | grad {node.grad:.4f}}}",
            shape="record",
        )
        if node._op:
            # If this value is a result of some operation, create an op node for it...
            graph.node(name=node_id + node._op, label=node._op)
            # ... and connect this node to it.
            graph.edge(node_id + node._op, node_id)

    # Finally, connect all nodes corresponding to the edges from trace.
    for prev_node, curr_node in edges:
        # Connect prev_node to the op node of curr_node.
        graph.edge(str(id(prev_node)), str(id(curr_node)) + curr_node._op)

    return graph


def topological_sort(node: Node) -> list[Node]:
    """Builds a topological ordering of the (forward) computational graph.

    Args:
        node: The node whose forward computational graph we consider.

    Returns:
        The list of nodes in the computational graph, sorted by their topological
        ordering. The first node in the list is a leaf of the forward graph,
        thus, it is a valid first element in the forward graph's topological ordering.

    """
    topological_ordering = []
    visited = set()

    def helper(node: Node) -> None:
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                helper(child)
            topological_ordering.append(node)

    helper(node)

    return topological_ordering


def apply_random_seed(random_seed: int) -> None:
    """Sets seed to ``random_seed`` in ``random``, ``numpy`` and ``torch``."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
