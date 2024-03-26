import numpy as np
import sys
import os
import sources.simulation as sim
import matplotlib.pyplot as plt
import random
import json
import multiprocessing as mp


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
            expn = np.exp(- n)
            return expn / np.square(1 + expn)



class Cost:
    class MeanSquaredError:
        def compute(output, target):
            return np.sum(np.square(output - target)) / output.shape[0]

        def deriv(output, target):
            return 2 * (output - target) / output.shape[0]



class Layer:
    def new(input_size, n_count, activation, learning_rate, std_dev):
        layer = Layer()
        layer.gradients = np.zeros(shape=(input_size + 1, n_count))
        layer.gradient_count = 0
        layer.params = np.random.normal(scale=std_dev, size=(input_size + 1, n_count))
        layer.activation = activation
        layer.learning_rate = learning_rate
        return layer

    def from_file(filename):
        layer = Layer()
        with open("%s.meta.json" % filename, 'r') as f:
            metadata = json.load(f)
        layer.activation = getattr(Activation, metadata["activation"])
        layer.learning_rate = metadata["learning_rate"]
        layer.params = np.load("%s.npy" % filename)
        layer.gradients = np.zeros_like(layer.params)
        layer.gradient_count = 0
        return layer

    def save_to(self, filename):
        metadata = {
            "activation": self.activation.__name__,
            "learning_rate": self.learning_rate
        }
        with open("%s.meta.json" % filename, 'w') as f:
            json.dump(metadata, f)
        np.save("%s.npy" % filename, self.params, allow_pickle=False)  # disallow pickle to avoid cross-platform issues

    def add_gradient(self, grad):
        self.gradients = np.add(self.gradients, grad)
        self.gradient_count += 1

    def feed_forward(self, input_data, log_activations=False):
        input_data = np.c_[input_data, np.ones(input_data.shape[0])]  # add a column of ones to take account for the bias
        s = np.matmul(input_data, self.params)
        return self.activation.activate(s), s

    def apply_gradients(self):  # just apply backpropagation
        if self.gradient_count == 0:
            print('no gradients to apply')
            return
        self.params += -self.learning_rate * (self.gradients / self.gradient_count)
        self.gradients = np.zeros_like(self.gradients)
        self.gradient_count = 0

    def previous_layer_deriv(self, raw_output_logs, layer_deriv):  # compute deriv for the previous layer (propagate)
        drond_activation = self.activation.deriv(raw_output_logs)
        full_local_back = drond_activation * layer_deriv
        return np.matmul(full_local_back, self.params[:self.params.shape[0]-1, :self.params.shape[1]].transpose())

    def full_backprop(self, input_logs, raw_output_logs, layer_deriv):  # apply backpropagation and compute for the previous layer (propagate)
        # here i'm not using the previous method to avoid computing the same thing twice
        drond_activation = self.activation.deriv(raw_output_logs)
        full_local_back = drond_activation * layer_deriv  # = bias gradient
        weights_deriv = np.matmul(input_logs.transpose(), full_local_back)
        full_params_deriv = np.r_[weights_deriv, np.zeros_like(full_local_back)]
        self.add_gradient(full_params_deriv)
        return np.matmul(full_local_back, self.params[:self.params.shape[0]-1, :self.params.shape[1]].transpose())



class Network:
    def new(layers, cost_func):
        net = Network()
        net.layers = layers
        net.cost_func = cost_func
        return net

    def from_file(filepath):
        with open(os.path.join(filepath, "meta.json"), 'r') as f:
            metadata = json.load(f)
        net = Network()
        net.cost_func = getattr(Cost, metadata['cost'])
        net.layers = []
        for l_name in metadata['layers']:
            net.layers.append(Layer.from_file(os.path.join(filepath, l_name)))
        return net

    def save_to(self, filepath):
        """
        str filepath: the path to the directory where the model will be contained
        """
        if os.path.exists(filepath):
            for item in os.listdir(filepath):
                os.remove(os.path.join(filepath, item))
        else:
            os.makedirs(filepath)
        metadata = {
            "cost": self.cost_func.__name__,
            "layers": []
        }
        for n, layer in enumerate(self.layers):
            layer_name = "layer_%i" % n
            layer.save_to(os.path.join(filepath, layer_name))
            metadata["layers"].append(layer_name)
        with open(os.path.join(filepath, "meta.json"), 'w') as f:
            json.dump(metadata, f)

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
        return self.cost_func.compute(input_data, target_data)

    def apply_learning(self):
        for layer in self.layers:
            layer.apply_gradients()



class TrainableParam:
    """
    some functions i use : https://www.desmos.com/calculator/srhiwurbop
    """
    def __init__(self, min, max, init=None, isint=False):
        self.min = min
        self.max = max
        self.delta = self.max - self.min
        self.isint = isint
        if init is not None:
            self._value = init
        else:
            self._value = random.uniform(min, max)

    @property
    def value(self):
        if self.isint:
            return round(self._value)
        return self._value

    @value.setter
    def value(self, value):
        self.value = min(self.max, max(self.min, value))

    def get_squished_value(self):
        return -np.log(self.delta/(self.value - self.min)-1)

    def get_linear_value(self):
        return (self.value - self.min)/self.delta

    def sig_inv_deriv(self, x):
        d = x - self.min
        return self.delta / (np.square(d) * ((D) / (d) - 1))

    def update(self, delta, value):
        self.value = self._value + self.sig_inv_deriv(value) * delta


class TrainableSim:
    def __init__(self, grid_size, pop_size, network_layers, data, learning_rate, max_time=100_000):
        self.grid_size = grid_size
        self.pop_size = pop_size
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.max_time = max_time
        self._data = data
        self.trainable_count = 0
        self.trainable_params = []
        for v in data.values():
            for sub_v in v:
                if type(sub_v) is TrainableParam:
                    self.trainable_count += 1
                    self.trainable_params.append(sub_v)

    @property
    def data(self):
        data = {}
        for k, v in self._data.items():
            data[k] = []
            for value in v:
                if type(value) is TrainableParam:
                    data[k].append(value.value)
                else:
                    data[k].append(value)
        return data

    @property
    def network_data(self):
        return [v.get_linear_value() for v in self.trainable_params]

    def update_params(self, deltas):
        for param, delta in zip(self.trainable_params, deltas):
            param.update(delta * self.learning_rate)

    def external_sim(self, queue):
        n = 0
        sim = sim.Simulation(self.grid_size, self.pop_size, self.network_layers, self.data)
        while sim.step():
            n += 1
            if n > self.max_time:
                break
        queue.put([self.network_data, np.array([n])])



def sample_data(trainable_sims):
    data = []
    ma_grosse_queue = mp.Queue()  # no offence
    for p in trainable_sims:
        mp.Process(p.external_sim(ma_grosse_queue)).start()
    for _ in range(len(train_params)):
        data.append(ma_grosse_queue.get())
    return data


def train_params(iterations, trainable_sims_list):
    for i in iterations:
        data = sample_data


if __name__ == "__main__":

    # do test things

    layers = [
        Layer.new(2, 5, Activation.Sigmoid, 1, 5),
        Layer.new(5, 5, Activation.Sigmoid, 1, 5),
        Layer.new(5, 1, Activation.Sigmoid, 1, 5),
    ]
    net = Network.new(layers, Cost.MeanSquaredError)
    data = np.random.normal(scale=10, size=(2,))
    print(net.feed_forward(data))
    net.save_to('./mynet')
    net2 = Network.from_file('./mynet')
    print(net2.feed_forward(data))

    # and just exit (temporarely)
    sys.exit()


    print(sys.argv)
    if len(sys.argv) < 2: # mauvais arguments
        print("erreur, mauvais arguments\n`%s -h` ou `%s --help` pour afficher l'aide" % (sys.argv[0], sys.argv[0]))
        sys.exit()

    if "-h" in sys.argv or "--help" in sys.argv:
        print(help_msg)
        sys.exit()
