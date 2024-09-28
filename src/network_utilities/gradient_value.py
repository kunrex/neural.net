from abc import ABC, abstractmethod

class GradientValue(ABC):
    def __init__(self):
        self._gradient_input = []

    def push_gradient(self, value):
        self._gradient_input.append(value)

    @abstractmethod
    def set_gradient(self):
        return