from node import Node
from src.network_utilities.iterator import Iterator

class Layer:
    def __init__(self, node_count):
        self.__node_count = node_count
        self.__nodes = [Node() for i in range(0, self.__node_count)]

    def __iter__(self):
        return Iterator(self.__nodes)

    def initialise(self, nodes):
        self.__nodes = nodes

    def node_count(self):
        return self.__node_count

    def get_node(self, index):
        if 0 <= index < self.__node_count:
            return self.__nodes[index]

        return False

    def set_node(self, index, value):
        if 0 <= index < self.__node_count:
            return self.__nodes[value].set_state(value)

        return False