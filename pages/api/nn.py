# The ANN structure

import random
import math

MLP_LAYOUT = [784, 100, 100, 10]  # 784 pixels input, 10 outputs (0, 1,...,9), arbitrary hidden layers
ALPHA = 0.1  # leaky ReLU slope parameter
LEARNING_RATE = 0.01
BATCH_SIZE = 1


# Basic structure for neuron values + constants related to training & the activation f'n
class Value:
    
    def __init__(self, val, grad=0.0):
        self.val = val
        self.grad = grad
    
    def __repr__(self):
        return "Value obj: " + str(self.val)

    def __add__(self, other):
        return Value(self.val + other.val, self.grad)

    def __mul__(self, other):
        return Value(self.val * other.val, self.grad)

    def __gt__(self, other):
        return self.val > other.val

    def reLU(self):
        return Value(max(0, self.val), self.grad)

    # leaky ReLU. slope parameter chosen arbitrarily
    def LreLU(self):
        if (self.val < 0):
            return Value(ALPHA * self.val, self.grad)
        return Value(self.val, self.grad)


class Neuron:

    def __init__(self, n, nout):
        #self.weights = [Value(random.uniform(-0.3,0.3)) for i in range(nout)]  # random uniform init
        std_dv = (2 / n) ** 0.5
        self.weights = [Value(random.gauss(0, std_dv)) for i in range(nout)]  # He Kaiming init
        self.val = Value(0.0)
        self.bias = Value(0.0)


class Layer:

    def __init__(self, n, nout):
        self.neurons = [Neuron(n, nout) for i in range(n)]       
        

class MLP:

    def __init__(self, MLP_LAYOUT):
        self.layers = [Layer(size, next_size) 
                       for size, next_size in zip(MLP_LAYOUT, MLP_LAYOUT[1:] + [0])]


    # currently O(n^3) T∆T
    def forward(self, inpt):        
        # initialize neuron values to 0 before beginning
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.val.val = 0.0
    
        for i in range(len(inpt)):  # loading first layer of mlp
            self.layers[0].neurons[i].val.val = inpt[i]
    
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            for neuron2, n in zip(layer2.neurons, range(len(layer2.neurons))):
                neuron2.val += neuron2.bias;  # load with its bias
                for neuron1 in layer1.neurons:  # load with val*weight of each prev neuron
                    neuron2.val += neuron1.val * neuron1.weights[n]
                neuron2.val = neuron2.val.LreLU()  # commpression/activation f'n   
    
    # returns the result of the last forward pass
    def result(self):
        last_layer = self.layers[-1]
        x = 0
            
        for i in range(1, len(last_layer.neurons)):
            if (last_layer.neurons[i].val > last_layer.neurons[x].val):
                x = i
        return x

    # cross-entropy loss
    def lossCE(self, desired):
        neurons = self.layers[-1].neurons
        x = max([neurons[i].val.val for i in range(10)])
        denom = sum([math.exp(neurons[i].val.val - x) for i in range(10)])
        softmax = math.exp(neurons[desired.index(1)].val.val - x) / denom
        return -math.log(softmax)
    
    # previously there was a v1
    def backward_v2(self):
        # copy-paste of part 3 of backward()
        for layer2, layer1 in zip(reversed(self.layers[:-1]), reversed(self.layers[1:])):
            for neuron2 in layer2.neurons:
                # keep in mind; # of weights in any neuron of layer1 = # of neurons in layer2
                for neuron1, n in zip(layer1.neurons, range(len(layer1.neurons))):                    
                    x = 1  # x represents d(neuron1.val)/d(neuron1.val before ReLU) = dsig/da
                    y = 1
                    if (neuron1.val.val < 0): x = ALPHA  #LReLU
                    if (neuron2.val.val < 0): y = ALPHA
              
                    neuron2.weights[n].grad = neuron1.val.grad * neuron2.val.val * x  # dC/dw = dC/dsig * dsig/da * da/dw
                    neuron2.val.grad += neuron1.val.grad * neuron2.weights[n].val * x  # dC/da = sum(dC/dw * dw/da)
                    neuron2.bias.grad += neuron1.val.grad * x * neuron2.weights[n].val * y
    
    # updates mlp using the gradients of each value
    def update(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                # update bias
                neuron.bias.val -= LEARNING_RATE * neuron.bias.grad
                # update weights
                for weight in neuron.weights:
                    weight.val -= LEARNING_RATE * weight.grad
                    
    # initializes first gradient layer based on softmax and categorical CE loss
    def initgrads(self, desired_output):
        val_gradient = 0.0
        
        neurons = self.layers[-1].neurons
        x = max([neurons[j].val.val for j in range(10)])  # to avoid math overflow error, shifts exp values towards 0
        for i in range(len(desired_output)):
            denom = sum([math.exp(neurons[j].val.val - x) for j in range(10)])
            if desired_output[i] == 0:  # for non-target logits, grad is softmax of this logit
                val_gradient = math.exp(neurons[i].val.val - x) / denom
            else:  # for target logit, grad is [softmax of this logit - 1]
                val_gradient = (math.exp(neurons[i].val.val - x) / denom) - 1
            neurons[i].val.grad += val_gradient
            
            if neurons[i].val.val < 0:
                neurons[i].bias.grad += val_gradient * ALPHA
            else: neurons[i].bias.grad += val_gradient

    # used for batch training
    def divide_first_layer_grads(self):
        for neuron in self.layers[-1].neurons:
            neuron.val.grad /= BATCH_SIZE
            neuron.bias.grad /= BATCH_SIZE

    # zeroes out all gradients
    def zero_grads(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.val.grad = 0.0
                neuron.bias.grad = 0.0
                for weight in neuron.weights:
                    weight.grad = 0.0