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
    def sample(self) -> dict[object, object]:
        """Returns a sample drawn from the implemented probability distribution."""
    @abstractmethod
    def max_inference(self) -> float:
        """Returns the maximum probability/density of the implemented probability distribution."""
    @abstractmethod
    def mpe(self) -> dict[object, object]:
        """Returns the most likely value of the implemented probability distribution."""
    @abstractmethod
    def equals(self, other: object) -> bool:
        """Returns True if the current object is identical to the other object."""


class OffsetLeaf(PCLeaf):
    """Leaf with empty scope and a probability/density of 1.0."""

    def __init__(self):
        PCLeaf.__init__(self, set())

    def inference(self, instance: dict) -> float:
        return 1.0

    def sample(self) -> dict[object, object]:
        return {}

    def max_inference(self) -> float:
        return 1.0

    def mpe(self) -> dict[object, object]:
        return {}

    def equals(self, other: object) -> bool:
        return isinstance(other, OffsetLeaf)


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
        if s not in inst or self.value != inst[s]:
            return 0.0
        return 1.0

    def sample(self) -> dict[object, object]:
        s, = self.scope
        return {s: self.value}

    def max_inference(self) -> float:
        return 1.0

    def mpe(self) -> dict[object, object]:
        s, = self.scope
        return {s: self.value}

    def equals(self, other: object) -> bool:
        if isinstance(other, ValueLeaf):
            return self.scope == other.scope and self.value == other.value
        return False
