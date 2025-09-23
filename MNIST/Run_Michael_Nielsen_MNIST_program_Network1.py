import MNIST_Loader
training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10])
net.SGD(training_data, epochs=3, mini_batch_size=10, eta=3.0, test_data=test_data)
print()
#  Save the Weights
w2 = net.weights[0]
b2 = net.biases[0]
print('w2.shape =', w2.shape)
print('b2.shape =', b2.shape)
print()
w3 = net.weights[1]
b3 = net.biases[1]
print()
print('w3.shape =', w3.shape)
print('b3.shape =', b3.shape)
print()
print()

data = dict({'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3})

#  See https://www.askpython.com/python/examples/save-data-in-python
import pickle

class MyClass():
    def __init__(self, param, weights = [], biases = []):
        self.param = param
        self.weights = weights
        self.biases = biases
        
def save_object(obj, picklefile_name = 'Net.pickle'):
    try:
        with open(picklefile_name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
my_net_obj = MyClass(data, [w2, w3], [b2, b3])
save_object(my_net_obj, 'Net.pickle')