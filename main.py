import mnist
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

input_layer_size = 784
hidden_layer_size = 100
num_labels = 10
regularization_parameter = 0.4
alpha = 0.5

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

# train_images = train_images[0:10000, :]
# train_labels = train_labels[0:10000]
# test_images = test_images[0:2000, :]
# test_labels = test_labels[0:2000]

train_images = np.transpose(train_images)
test_images = np.transpose(test_images)
train_images = train_images / 255
test_images = test_images / 255

row_add = np.ones(len(train_images[0]))
train_images = np.vstack((row_add, train_images))

iteration_number = 0

# theta_1 = np.zeros((50, 785))
# theta_2 = np.zeros((10, 51))
# print(np.shape(train_labels))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def random_initialization(l_in, l_out):
    # return np.random.rand(l_out, l_in + 1) * 2 - 1
    return np.random.randn(l_out,l_in + 1) * np.sqrt(2/l_in + 1)


def cost_function(nn_parameters, x, y):
    global iteration_number
    print('Iteration: ' + str(iteration_number))
    iteration_number = iteration_number + 1
    theta_layer_1 = nn_parameters[0:np.size(theta_1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta_layer_2 = nn_parameters[np.size(theta_1):len(nn_parameters)].reshape(num_labels, hidden_layer_size + 1)
    theta_1_grad = np.zeros(np.shape(theta_layer_1))
    theta_2_grad = np.zeros(np.shape(theta_layer_2))

    m = len(x[0])

    temp = sigmoid(theta_layer_1 @ x)
    row_to_add = np.ones(m)
    temp = np.vstack((row_to_add, temp))

    y_pred = sigmoid(theta_layer_2 @ temp)
    y_binary = np.zeros((num_labels, len(y)))
    to_print = np.hstack((np.transpose(y_pred), np.transpose(y_binary)))
    # with np.printoptions(threshold=np.inf):
        # print(np.transpose(y_pred))
    accuracy = (np.sum(y_pred == y_binary) / m) * 10
    # print('Accuracy1: ' + str(np.sum(y_pred == y_binary)))

    for i in range(len(y)):
        y_binary[y[i], i] = 1

    accuracy = (np.sum(y_pred == y_binary) / m) * 10
    # print('Accuracy: ' + str(accuracy))
    temp_1 = theta_layer_1
    temp_2 = theta_layer_2

    global regularization_parameter

    cost = (-1 * (np.sum((np.multiply(y_binary, np.log(y_pred))) + (np.multiply((1.0 - y_binary), np.log(1.0 - y_pred)))))) / m
    cost += (regularization_parameter / (2 * m)) * (np.square(theta_layer_1)[:, 1:len(theta_layer_1[0])].sum() + np.square(theta_layer_2)[:, 1:len(theta_layer_2[0])].sum())
    # print('Cost: ' + str(cost))

    d2 = y_pred - y_binary
    d1 = np.multiply((np.transpose(theta_layer_2) @ d2), temp, (1 - temp))

    theta_2_grad = ((d2 @ np.transpose(temp)) + regularization_parameter * theta_layer_2) / m
    # theta_2_grad = (d2 @ np.transpose(temp))
    theta_2_grad[:, 0] -= (regularization_parameter * theta_layer_2[:, 0]) / m
    theta_1_grad = ((d1[1:len(d1), :] @ np.transpose(x)) + regularization_parameter * theta_layer_1) / m
    # theta_1_grad = (d1[1:len(d1), :] @ np.transpose(x))
    theta_1_grad[:, 0] -= (regularization_parameter * theta_layer_1[:, 0]) / m

    nn_grad = np.vstack((theta_1_grad.reshape(np.size(theta_1_grad), 1), theta_2_grad.reshape(np.size(theta_2_grad), 1)))

    nn_grad = nn_grad.flatten()
    print(cost)
    return cost, nn_grad


theta_1 = random_initialization(input_layer_size, hidden_layer_size)
theta_2 = random_initialization(hidden_layer_size, num_labels)

initial_nn_parameters = np.vstack((theta_1.reshape(np.size(theta_1), 1), theta_2.reshape(np.size(theta_2), 1)))
initial_nn_parameters = initial_nn_parameters.flatten()
print(np.shape(initial_nn_parameters))

nn_parameters = np.vstack((theta_1.reshape(np.size(theta_1), 1), theta_2.reshape(np.size(theta_2), 1)))
nn_parameters = nn_parameters.flatten()
print(np.shape(nn_parameters))

error = cost_function(nn_parameters, train_images, train_labels)
# theta_1_gradients = nn_gradients[0:np.size(theta_1)].reshape(hidden_layer_size, input_layer_size + 1)
# theta_2_gradients = nn_gradients[np.size(theta_1):len(nn_gradients)].reshape(num_labels, hidden_layer_size + 1)

# cost_compiler = np.array(error)

# for i in range(500):
    # print(i, end='\r')
    # cost, gradient = cost_function(nn_parameters, train_images, train_labels)
    # alpha = 1.5 / cost
    # nn_parameters = nn_parameters - alpha * gradient
    # np.append(cost_compiler, cost)

nn_cost_function = lambda p: cost_function(p, train_images, train_labels)
result_of_minimization = opt.minimize(nn_cost_function, initial_nn_parameters, jac=True, method='CG', bounds=None, tol=None, callback=None, options={'gtol': 0.0001})
nn_parameters = result_of_minimization.x
cost = cost_function(nn_parameters, train_images, train_labels)
theta_1_params = nn_parameters[0:np.size(theta_1)].reshape(hidden_layer_size, input_layer_size + 1)
theta_2_params = nn_parameters[np.size(theta_1):len(nn_parameters)].reshape(num_labels, hidden_layer_size + 1)

# print('Cost: ' + str(cost))
m = len(test_images[0])
row_to_add = np.ones(m)
test_images = np.vstack((row_to_add, test_images))
x = np.transpose(test_images)

temp = sigmoid(theta_1_params @ np.transpose(x))
row_to_add = np.ones(len(temp[0]))
temp = np.vstack((row_to_add, temp))
# print(np.shape(temp))
# with np.printoptions(threshold=np.inf):
    # print(np.transpose(temp))

# y_pred = np.zeros(np.shape(theta_2_params @ temp))
# print(theta_2_params[0] @ temp[:, 0])
# print(theta_2_params[1] @ temp[:, 1])
# with np.printoptions(threshold=np.inf):
    # print(np.transpose(y_pred))
y_pred = sigmoid(theta_2_params @ temp)
y_pred = np.transpose(y_pred)
# y_pred = np.around(y_pred)
# with np.printoptions(threshold=np.inf):
    # print(y_pred)
dummy = np.zeros(np.shape(test_labels))
# print(len(y_pred))
# print(len(y_pred[0]))
# print(np.shape(dummy))
for i in range(len(y_pred)):
    dummy[i] = np.argmax(y_pred[i])
    # for j in range(len(y_pred[0])):
        # if y_pred[i, j] == 1:
            # dummy[i] = j
with np.printoptions(threshold=np.inf):
    print(np.count_nonzero(test_labels - dummy))
# print(np.shape(test_labels))
# accuracy = (np.sum(y_pred == train_labels) / np.size(train_labels)) * 100
# print('Accuracy: ' + str(accuracy))


# x_axis_values = np.arange(1001)

# plt.plot(x_axis_values, cost_compiler)
# plt.xlabel('Iterations')
# plt.ylabel('Cost')
# plt.title('MNIST')
# plt.show()

# with np.printoptions(threshold=np.inf):
# print(test_labels)
