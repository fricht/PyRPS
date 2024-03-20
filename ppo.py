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



class Layer:
    def new(input_size, n_count, activation, learning_rate, std_dev):
        layer = Layer()
        layer.gradients = np.zeros(shape=(n_count, input_size + 1))
        layer.params = np.random.normal(scale=std_dev, size=(input_size + 1, n_count))
        layer.activation = activation
        layer.learning_rate = learning_rate
        return layer

    def feed_forward(self, input_data, log_activations=False):
        input_data = np.c_[input_data, np.ones(input_data.shape[0])]  # add a column of ones to take account for the bias
        s = np.matmul(input_data, self.params)
        return self.activation.activate(s), s

    def apply_gradients(self): # just apply backpropagation
        self.params += self.learning_rate * self.gradients
        self.gradients = np.zeros_like(self.gradients)

    def previous_layer_deriv(self, input_logs, raw_output_logs, layer_deriv): # compute deriv for the previous layer (propagate)
        drond_activation = self.activation.deriv(raw_output_logs)
        full_back = drond_activation * layer_deriv
        return np.matmul(drond_activation, self.params[:self.params.shape[0]-1, :self.params.shape[1]].transpose())

    def full_backprop(self, input_logs, layer_deriv): # apply backpropagation and compute for the previous layer (propagate)
        # here i'm not using the previous method to avoid computing the same thing twice
        drond_activation = self.activation.deriv(input_logs.transpose())
        weights_deriv = np.matmul(input_logs.transpose(), drond_activation)
        full_deriv = np.r_[weights_deriv, drond_activation]
        self.gradients = np.add(self.gradients, full_deriv)
        return np.matmul(drond_activation, self.params[:self.params.shape[0]-1, :self.params.shape[1]].transpose())



class Network:
    def new(layers):
        net = Network()
        net.layers = layers
        net.activ_logs = []
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
            logs.append(input_data)
            input_data, _ = layer.feed_forward(input_data)
        deriv = np.ones_like(input_data)
        for layer, activ in zip(reversed(self.layers), reversed(logs)):
            deriv = layer.previous_layer_deriv(activ, deriv)
        return deriv[0]

    def compute_backpropagation(self):
        for data in self.activ_logs:
            deriv = data[-1]
            for layer, activ in zip(reversed(self.layers), reversed(data)[1::]):
                deriv = layer.full_backprop(activ, deriv)
        self.data = []



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
    net = Network.new([Layer.new(1, 10, Activation.Sigmoid, 0.1, 1), Layer.new(10, 2, Activation.LReLU, 0.1, 1)])
    d = np.array([1])
    print(net.feed_forward(d))
    for _ in range(6):
        print('---')
        deriv = net.input_deriv_only(d)
        print(deriv)
        d = np.add(d, deriv * 10)
        print(d)
        print(net.feed_forward(d))
        print('###')
    sys.exit()
    # then exit

    print(sys.argv)
    if len(sys.argv) < 2: # mauvais arguments
        print("erreur, mauvais arguments\n`%s -h` ou `%s --help` pour afficher l'aide" % (sys.argv[0], sys.argv[0]))
        sys.exit()

    if "-h" in sys.argv or "--helf" in sys.argv:
        print(help_msg)
        sys.exit()
