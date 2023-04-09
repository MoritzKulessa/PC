from abc import ABC, abstractmethod


class PCNode(object):
    """Base class for a node."""
    def __init__(self, scope: set = None):
        self.scope = set() if scope is None else scope


class PCInnerNode(PCNode):
    """Base class for an inner node."""
    def __init__(self, children: list, scope: set = None):
        PCNode.__init__(self, scope)
        self.children = children


class PCSum(PCInnerNode):
    """Sum unit (mixture model)."""
    def __init__(self, children: list, weights: list, scope: set = None):
        PCInnerNode.__init__(self, children, scope)
        assert (len(weights) == len(children))
        self.weights = weights


class PCProduct(PCInnerNode):
    """Product unit (factorized model)."""
    def __init__(self,
                 children: list,
                 scope: set = None):
        PCInnerNode.__init__(self, children, scope)


class PCLeaf(PCNode, ABC):
    """Distribution unit (probability distribution)"""
    def __init__(self, scope: set):
        PCNode.__init__(self, scope)

    @abstractmethod
    def inference(self, instance: dict) -> float:
        """Returns the probability/density for the given instance."""
    @abstractmethod
    def sample(self) -> dict:
        """Returns a sample drawn from the implemented probability distribution."""
    @abstractmethod
    def max_prob(self) -> float:
        """Returns the maximum probability/density of the implemented probability distribution."""
    @abstractmethod
    def max_value(self) -> dict:
        """Returns the most likely value of the implemented probability distribution."""


class OffsetLeaf(PCLeaf):
    """Leaf with empty scope and a probability/density of 1.0."""

    def __init__(self):
        PCLeaf.__init__(self, set())

    def inference(self, instance: dict) -> float:
        return 1.0

    def sample(self) -> dict:
        return {}

    def max_prob(self) -> float:
        return 1.0

    def max_value(self) -> dict:
        return {}


class ValueLeaf(PCLeaf):
    """
    Leaf which represents the probability distribution for a single value for an attribute.
    Since only one value is allowed, it always has a probability of 1.0.
    """

    def __init__(self, scope: set, value: object):
        PCLeaf.__init__(self, scope)
        assert (len(scope) == 1)
        self.value = value

    def inference(self, inst: dict) -> float:
        s, = self.scope
        if self.value == inst[s]:
            return 1.0
        return 0.0

    def sample(self) -> dict:
        s, = self.scope
        return {s: self.value}

    def max_prob(self) -> float:
        return 1.0

    def max_value(self) -> dict:
        s, = self.scope
        return {s: [self.value]}
