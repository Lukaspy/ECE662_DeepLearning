import pickle
 
class MyClass():
    def __init__(self, param):
        self.param = param
 
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
 
data = load_object("Net.pickle")
dict_data = data.param
print('type(data) =', type(data))
print('type(data_dict =',type(dict_data)) 
print('list(dict_data) =',list(dict_data))
#print(obj.param)
#print(isinstance(obj, MyClass))

w_2 = dict_data['w2']
b_2 = dict_data['b2']
w_3 = dict_data['w3']
b_3 = dict_data['b3']

w2 = data.weights[0]
w3 = data.weights[1]
b2 = data.biases[0]
b3 = data.biases[1]

import matplotlib.pyplot as plt
rows, cols = 5, 6
print()
print("Plot the W2 weights")
print

fig, axs = plt.subplots(rows, cols)
plt.subplots_adjust(hspace = 0.8)
for j in range(30):
    h1_image = w2[j].reshape(28,28)
    print()
    print("The weights into h{0} have image ".format(j))
    ax = axs.flat[j]
    im = ax.imshow(h1_image,cmap='BuPu', origin='upper')
    ax.set_title("h{0} image".format(j))
#    plt.xticks(())
#    plt.yticks(())
#    print("plt.show():")
    print()
    if j==1: fig.colorbar(im,ax=axs, fraction=0.03, pad=0.02)
fig.suptitle("W2 weights local machine")
plt.show()


print()
print("Plot the W3 weights.")
print()
rows, cols = 5, 2
fig, axs = plt.subplots(rows, cols)
plt.subplots_adjust(hspace = 0.8)
for j in range(10):
    print("The weights into output {0} are".format(j))
    ax = axs.flat[j]
    ax.plot(w3[j],'o')
    ax.set_title("Output {0} weights".format(j))
    print()

fig.suptitle("W3 weights local machine")
plt.show()
