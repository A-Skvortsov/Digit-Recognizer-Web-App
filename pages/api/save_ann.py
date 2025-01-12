# Code for saving the ANN (its weights and biases) to external csv files
import csv
import nn
EMPTY = "          "  # for formatting purposes
B_SAVETO = '1_all_biases.csv'
W_SAVETO = '1_all_weights.csv'
N_SAVETO = '1_all_neurons.csv'

B_LOAD = '0_all_biases.csv'
W_LOAD = '0_all_weights.csv'

"""
Saving all ANN info to csv for debugging and general analysis
Saves:
    - current weights and biases
    - all current gradients
    - all current neuron values (as a result of last forward pass)
"""
def save():
    save_biases()
    save_neurons()
    save_weights()
    
"""
Loading saved ANN info (weights and biases) from csv.
"""
def load():
    load_biases()
    load_weights()


""" ================================================================================================
====================================================================================================
IMPLEMENTATIONS BELOW ==============================================================================
====================================================================================================
================================================================================================ """
    
    
def save_biases():
    with open(B_SAVETO, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['B0', 'B1', 'B2', 'B3', '', 'BG0', 'BG1', 'BG2', 'BG3'])
    
        for i in range(max([len(mlp.layers[i].neurons) for i in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
            row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
            for j in range(len(mlp.layers)):  # biases
                try: row[j] = mlp.layers[j].neurons[i].bias.val
                except: row[j] = EMPTY
            row[len(mlp.layers)] = EMPTY  # space column for readability
            for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # bias gradients
                try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].bias.grad
                except: row[j] = EMPTY
            writer.writerow(row)

def save_neurons():
    with open(N_SAVETO, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['V0', 'V1', 'V2', 'V3', '', 'VG0', 'VG1', 'VG2', 'VG3'])
    
        for i in range(max([len(mlp.layers[i].neurons) for i in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
            row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
            for j in range(len(mlp.layers)):  # values
                try: row[j] = mlp.layers[j].neurons[i].val.val
                except: row[j] = EMPTY
            row[len(mlp.layers)] = EMPTY  # space column for readability
            for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # value gradients
                try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].val.grad
                except: row[j] = EMPTY
            writer.writerow(row)
            
def save_weights():
    with open(W_SAVETO, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['W0', 'W1', 'W2', 'W3', '', 'WG0', 'WG1', 'WG2', 'WG3'])

        for i in range(max([len(mlp.layers[x].neurons) for x in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
            row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
            for w in range(max([len(mlp.layers[x].neurons[0].weights) for x in range(len(mlp.layers))])):  # 'for weight in layer with most weights'
                for j in range(len(mlp.layers)):  # weights
                    try: row[j] = mlp.layers[j].neurons[i].weights[w].val
                    except: row[j] = EMPTY
                row[len(mlp.layers)] = EMPTY  # space column for readability
                for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # weight gradients
                    try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].weights[w].grad
                    except: row[j] = EMPTY
                writer.writerow(row)
            writer.writerow([EMPTY for i in range(2 * len(mlp.layers) + 1)])
    
def load_biases(mlp):
    with open(B_LOAD, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.__next__()  # gets rid of top row labels
    
        for row in reader:
            for i in range(len(nn.MLP_LAYOUT)):
                if reader.line_num - 2 < nn.MLP_LAYOUT[i]:  # reader.line_num is at 2 upon first iteration of outer loop
                    mlp.layers[i].neurons[reader.line_num - 2].bias = nn.Value(float(row[i]))
                    
def load_weights(mlp):
    with open(W_LOAD, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.__next__()
    
        for n in range(max(nn.MLP_LAYOUT)):
            i = 0  # current weight of neuron n
            while i < max([len(mlp.layers[x].neurons[0].weights) for x in range(len(nn.MLP_LAYOUT))]):  # while i is less than max # of weights in any layer
                row = reader.__next__()
                for j in range(len(nn.MLP_LAYOUT)):  # j is current layer
                    if i < len(mlp.layers[j].neurons[0].weights) and n < len(mlp.layers[j].neurons):
                        mlp.layers[j].neurons[n].weights[i] = nn.Value(float(row[j]))
                i += 1
            reader.__next__()  # skips empty lines, which occur every 100 lines to indicate new neuron