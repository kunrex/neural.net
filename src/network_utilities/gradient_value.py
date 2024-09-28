from abc import ABC, abstractmethod

class GradientValue(ABC):
    def __iter__(self):
        self.__gradient_input = []

    def push_gradient(self, value):
        self.__gradient_input.append(value)

    @abstractmethod
    def set_gradient(self):
        return