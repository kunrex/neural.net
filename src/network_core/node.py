from src.network_utilities.iterator import Iterator
from src.network_utilities.gradient_value import GradientValue

class Node(GradientValue):
    def __init__(self):
        super().__init__()

        self.__state = 0
        self.__raw_state = 0

        self.__bias = 0

        self.__branch_in = []
        self.__branch_out = []

    def __iter__(self):
        return Iterator(self.__branch_in)

    def with_branch_in(self, branch_in):
        self.__branch_in.append(branch_in)

    def with_branch_out(self, branch_out):
        self.__branch_out.append(branch_out)

    def get_branch_in(self, branch_in):
        for branch in self.__branch_in:
            if branch.get_node_a() == branch_in:
                return branch

        return None

    def get_branch_out(self, branch_out):
        for branch in self.__branch_out:
            if branch.get_node_b() == branch_out:
                return branch

        return None

    def get_state(self):
        return self.__state

    def get_raw_state(self):
        return self.__raw_state

    def set_state(self, state, raw_state):
        if 0 <= state <= 1:
            self.__state = state
            self.__raw_state = raw_state
            return True

        return False

    def get_bias(self):
        return self.__bias

    def set_gradient(self):
        self.__bias += -sum(self._gradient_input) / len(self._gradient_input)
        self._gradient_input.clear()