import numpy as np
from abc import ABC, abstractmethod


class PCNode(object):
    def __init__(self, scope=None):
        self.scope = set() if scope is None else scope


class PCSum(PCNode):
    def __init__(self, weights, children, scope=None):
        PCNode.__init__(self, scope)
        self.weights = weights
        self.children = children


class PCProduct(PCNode):
    def __init__(self, children, scope=None):
        PCNode.__init__(self, scope)
        self.children = children


class PCLeaf(PCNode, ABC):
    def __init__(self, scope=None):
        PCNode.__init__(self, scope)

    @abstractmethod
    def prob(self, inst): pass
    @abstractmethod
    def sample(self): pass
    @abstractmethod
    def relate(self): pass
    @abstractmethod
    def max_prob(self): pass
    @abstractmethod
    def max_value(self): pass


class OffsetLeaf(PCLeaf):
    def __init__(self):
        PCNode.__init__(self)

    def prob(self, inst):
        return 1.0

    def sample(self):
        return {}

    def relate(self):
        return {}

    def max_prob(self):
        return 1.0

    def max_value(self):
        return {}


offset_leaf = OffsetLeaf()


class ValueLeaf(PCLeaf):
    def __init__(self, scope, value):
        PCLeaf.__init__(self, scope)
        assert (len(scope) == 1)    # Not implemented yet for multivariate distributions
        self.value = value

    def prob(self, inst):
        s, = self.scope
        if self.value == inst[s]:
            return 1.0
        return None

    def sample(self):
        s, = self.scope
        return {s: self.value}

    def relate(self):
        s, = self.scope
        return {(s, self.value): 1.0 }

    def max_prob(self):
        return 1.0

    def max_value(self):
        s, = self.scope
        return {s: [self.value]}


class CategoricalLeaf(PCLeaf):
    def __init__(self, scope, val_prob_dict):
        PCLeaf.__init__(self, scope)
        assert (len(scope) == 1)    # Not implemented yet for multivariate distributions
        assert (sum([prob for _, prob in val_prob_dict.items()]) == 1.0)
        self.val_prob_dict = val_prob_dict

    def prob(self, inst):
        s, = self.scope
        val = inst[s]
        if val not in self.val_prob_dict:
            return None
        return self.val_prob_dict[val]

    def sample(self):
        s, = self.scope
        values, probabilities = list(map(list, zip(*self.val_prob_dict.items())))
        return {s: np.random.choice(values, p=probabilities)}

    def relate(self):
        s, = self.scope
        values, probabilities = list(map(list, zip(*self.val_prob_dict.items())))
        return {(s, val): prob for val, prob in zip(values, probabilities)}

    def max_prob(self):
        return max(self.val_prob_dict.values())

    def max_value(self):
        s, = self.scope
        values, probabilities = list(map(list, zip(*self.val_prob_dict.items())))
        return {s: [values[index] for index in np.argwhere(probabilities == np.amax(probabilities)).flatten()]}


'''
class DictLeaf(PCLeaf):
    def __init__(self, d, scope=None):
        PCLeaf.__init__(self, scope)
        assert (isinstance(d, dict))
        self.d = d
        if len(self.d) == 0:
            self.n_instances = 0
        else:
            self.n_instances = sum([v for _, v in self.d.items()])

    def prob(self, inst):
        assert(len(self.scope) == 1)
        if self.n_instances == 0:
            return None
        s, = self.scope
        val = inst[s]
        if val not in self.d:
            return None
        assert (self.d[val] > 0)
        return self.d[val] / self.n_instances

    def sample(self):
        assert (len(self.scope) == 1)
        assert (self.n_instances > 0)
        assert (len(self.d) > 0)
        s, = self.scope
        keys, values = list(map(list, zip(*self.d.items())))
        return {s: np.random.choice(keys, p=np.array(values)/self.n_instances)}

    def relate(self):
        assert (len(self.scope) == 1)
        assert (self.n_instances > 0)
        assert (len(self.d) > 0)
        s, = self.scope
        return {str(s) + "=" + str(k): v for k, v in self.d.items()}

    def max_prob(self):
        assert (len(self.scope) == 1)
        assert (self.n_instances > 0)
        assert (len(self.d) > 0)
        keys, values = list(map(list, zip(*self.d.items())))
        index = np.argmax(values)
        return values[index]/self.n_instances

    def max_value(self):
        assert (len(self.scope) == 1)
        assert (self.n_instances > 0)
        assert (len(self.d) > 0)
        s, = self.scope
        keys, values = list(map(list, zip(*self.d.items())))
        index = np.argmax(values)
        return {s: keys[index]}
'''