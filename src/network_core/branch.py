import random

from src.network_utilities.gradient_value import GradientValue

class Branch(GradientValue):
    def __init__(self, node_a, node_b):
        super().__init__()

        self.__node_a = node_a
        self.__node_b = node_b

        self.__weight = random.randint(-4, 4)

    def get_node_a(self):
        return self.__node_a

    def get_node_b(self):
        return self.__node_b

    def weight_gradient(self, gradient):
        self.__weight += gradient

    def get_weight(self):
        return self.__weight

    def set_gradient(self):
        self.__weight += -sum(self._gradient_input) / len(self._gradient_input)
        self._gradient_input.clear()


