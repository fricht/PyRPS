import numpy as np
import sys
import os
import simulation as sim


class Activation:
    class Linear: # use as abstract class
        @np.vectorize
        def activate(n):
            return n

        @np.vectorize
        def deriv(_n):
            return 1

    class LReLU(Linear):
        a = 0.1
        @np.vectorize
        def activate(n):
            return max(Activation.LReLU.a * n, n)

        @np.vectorize
        def deriv(n):
            if n < 0:
                return Activation.LReLU.a
            return 1

    class Sigmoid(Linear):
        @np.vectorize
        def activate(n):
            return 1 / (1 + np.exp(- n))

        @np.vectorize
        def deriv(n):
            return np.exp(- n) / np.square(1 + np.exp(- n))



class Cost:
    class MeanSquaredError:
        def compute(output, target):
            return np.sum(np.square(output-target)) / output.shape[0]

        def deriv(output, target):
            return (2 * output - 2 * target) / 2



class Layer:
    def new(input_size, n_count, activation, learning_rate, std_dev):
        layer = Layer()
        layer.gradients = np.zeros(shape=(input_size + 1, n_count))
        layer.params = np.random.normal(scale=std_dev, size=(input_size + 1, n_count))
        layer.activation = activation
        layer.learning_rate = learning_rate
        return layer

    def feed_forward(self, input_data, log_activations=False):
        input_data = np.c_[input_data, np.ones(input_data.shape[0])]  # add a column of ones to take account for the bias
        s = np.matmul(input_data, self.params)
        return self.activation.activate(s), s

    def apply_gradients(self): # just apply backpropagation
        self.params += self.learning_rate * -self.gradients
        self.gradients = np.zeros_like(self.gradients)

    def previous_layer_deriv(self, raw_output_logs, layer_deriv): # compute deriv for the previous layer (propagate)
        drond_activation = self.activation.deriv(raw_output_logs)
        full_local_back = drond_activation * layer_deriv
        return np.matmul(full_local_back, self.params[:self.params.shape[0]-1, :self.params.shape[1]].transpose())

    def full_backprop(self, input_logs, raw_output_logs, layer_deriv): # apply backpropagation and compute for the previous layer (propagate)
        # here i'm not using the previous method to avoid computing the same thing twice
        drond_activation = self.activation.deriv(raw_output_logs)
        full_local_back = drond_activation * layer_deriv # = bias gradient
        weights_deriv = np.matmul(input_logs.transpose(), full_local_back)
        full_params_deriv = np.r_[weights_deriv, full_local_back]
        self.gradients = np.add(self.gradients, full_params_deriv)
        return np.matmul(full_local_back, self.params[:self.params.shape[0]-1, :self.params.shape[1]].transpose())



class Network:
    def new(layers, cost_func):
        net = Network()
        net.layers = layers
        net.cost_func = cost_func
        return net

    def feed_forward(self, input_data):
        input_data = np.array([input_data])
        for layer in self.layers:
            input_data, _ = layer.feed_forward(input_data)
        return input_data[0]

    def input_deriv_only(self, input_data):
        # this method only works for this use case !
        # it tries to maximize the output, which is not the usual thing a nn is for.
        input_data = np.array([input_data])
        logs = []
        for layer in self.layers:
            input_data, raw_log = layer.feed_forward(input_data)
            logs.append(raw_log)
        deriv = np.ones_like(input_data)
        for layer, log in zip(reversed(self.layers), reversed(logs)):
            deriv = layer.previous_layer_deriv(log, deriv)
        return deriv[0]

    def learn(self, input_data, target_data):
        input_data = np.array([input_data])
        logs = []
        for layer in self.layers:
            logs.append([input_data, None])
            input_data, raw_log = layer.feed_forward(input_data)
            logs[-1][1] = raw_log
        deriv = self.cost_func.deriv(input_data, target_data)
        for layer, log in zip(reversed(self.layers), reversed(logs)):
            deriv = layer.full_backprop(log[0], log[1], deriv)

    def apply_learning(self):
        for layer in self.layers:
            layer.apply_gradients()



def main(*args):
    pass


help_msg = """
utilisation :

    python3 ppo.py <fonction> [parametres ...]


fonctions :

    - ...

bref, à completer ...
"""


if __name__ == "__main__":
    # tests things
    net = Network.new([Layer.new(2, 2, Activation.Sigmoid, 0.1, 1), Layer.new(2, 1, Activation.Sigmoid, 0.1, 1)], Cost.MeanSquaredError)
    # XOR data
    data = [[np.array([0, 0]), np.array([0])], [np.array([1, 0]), np.array([1])], [np.array([0, 1]), np.array([1])], [np.array([1, 1]), np.array([0])]]
    #test
    print('### TESTING ###')
    for sample in data:
        print("for %s -> %s -> %s" % (sample[0], net.feed_forward(sample[0]), sample[1]))
    #learn
    print("### LEARNING ###")
    for _ in range(10000):
        for sample in data:
            net.learn(sample[0], sample[1])
    net.apply_learning()
    #test
    print('### TESTING ###')
    for sample in data:
        print("for %s -> %s -> %s" % (sample[0], net.feed_forward(sample[0]), sample[1]))
    sys.exit()
    # then exit

    print(sys.argv)
    if len(sys.argv) < 2: # mauvais arguments
        print("erreur, mauvais arguments\n`%s -h` ou `%s --help` pour afficher l'aide" % (sys.argv[0], sys.argv[0]))
        sys.exit()

    if "-h" in sys.argv or "--help" in sys.argv:
        print(help_msg)
        sys.exit()
