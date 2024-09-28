import math

from src.network_core.layer import Layer
from src.network_core.branch import Branch

def sigmoid(value):
    return 1 / (1 + math.exp(-value))

def sigmoid_derivative(value):
    value = math.exp(-value)
    return - value / (1 + value) ** 2

def cost_function(expected, output):
    cost = 0
    for i in range(0, len(output)):
        cost += (expected[i] - output[i]) ** 2

    return cost

class NeuralNetwork:
    __input_node = None
    __output_node = None

    def __init__(self, layers, nodes, node_in, node_out):
        self.__node_in = node_in
        self.__layers = [Layer(self.__node_in)]
        for i in range(0, layers):
            self.__layers.append(Layer(nodes))

        self.__node_out = node_out
        self.__layers.append(Layer(self.__node_out))

        self.__layerCount = len(self.__layers)
        self.__input_node = self.__layers[0]
        self.__output_node = self.__layers[-1]

    def initialise(self):
        for i in range(0, self.__layerCount - 1):
            layer_a = self.__layers[i]
            layer_b = self.__layers[i + 1]
            for node_a in layer_a:
                for node_b in layer_b:
                    branch = Branch(node_a, node_b)

                    node_a.with_branch_out(branch)
                    node_b.with_branch_in(branch)

        return self

    def train(self, batch):
        cost = 0
        for test in batch:
            for i in range(0, self.__node_in):
                self.__layers[0].set_node(i, test[i + 1])

            self.__front_propagate()
            result = []
            for node in self.__layers[-1]:
                result.append(node.get_state())

            self.__back_propagate(test[0])
            cost += cost_function(test[0], result)
            print("EXPECTED: {}; RESULT: {}".format( test[0], result))

        self.__back_propagate_push()

        print("BATCH COMPLETE")
        print("AVERAGE COST: {}".format(cost / len(batch)))

    def test(self, tests):
            for test in tests:
                for i in range(0, self.__node_in):
                    self.__layers[0].set_node(i, test[i + 1])

                self.__front_propagate()
                result = []
                for node in self.__layers[-1]:
                    result.append(node.get_state())

                print("EXPECTED: {}; RESULT: {}".format(test[0], result))

    def __front_propagate(self):
        for i in range(0, self.__layerCount - 1):
            layer_a = self.__layers[i]
            layer_b = self.__layers[i + 1]
            for end in layer_b:
                weighted = 0
                for start in layer_a:
                    branch = start.get_branch_out(end)
                    weighted += start.get_state() * branch.get_weight()

                end.set_state(sigmoid(weighted + end.get_bias()), weighted + end.get_bias())

    def __back_propagate(self, expected):
        state_gradients_prev = { }

        i = 0
        for node in self.__output_node:
            state_gradients_prev[node] = (2 * (node.get_state() - expected[i]))
            bias_gradient = state_gradients_prev[node] * sigmoid_derivative(node.get_raw_state())
            node.push_gradient(bias_gradient)

            for branch in node:
                branch.push_gradient(bias_gradient * branch.get_node_a().get_state())

            i += 1

        for i in range(self.__layerCount - 2, -1, -1):
            gradients = { }
            layer = self.__layers[i]

            for node in layer:
                state_gradient = 0
                for end in self.__layers[i + 1]:
                    branch = node.get_branch_out(end)
                    state_gradient += branch.get_weight() * sigmoid_derivative(node.get_raw_state()) * state_gradients_prev[end]

                bias_gradient = sigmoid_derivative(node.get_raw_state()) * state_gradient
                node.push_gradient(bias_gradient)

                for branch in node:
                    branch.push_gradient(bias_gradient * branch.get_node_a().get_state())

                gradients[node] = state_gradient

            state_gradients_prev.clear()
            state_gradients_prev.update(gradients)

    def __back_propagate_push(self):
        for i in range(0, self.__layerCount - 1):
            layer = self.__layers[i]
            for node in layer:
                node.set_gradient()
                for branch in node:
                    branch.set_gradient()

        for node in self.__output_node:
            node.set_gradient()

