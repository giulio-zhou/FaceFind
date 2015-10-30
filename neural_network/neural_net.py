import numpy as np
import scipy.optimize
from math import exp,log,tanh
from numpy import vectorize
from random import randint
# Initialize constants
learning_rate = 5e-3
lmbda = 0


def mse_grad(a3, y):
    if len(y.shape) == 2:
        y.shape = y.shape[0]
    return a3 - y


class Neural_net:
    def __init__(self, input_size, hidden_size,
                 output_size, theta1=None, theta2=None):
        # Allow for random initialization of weights if not provided
        if theta1 == None:
            self.theta1 = np.random.rand(input_size + 1, hidden_size) - 0.5
        else:
            self.theta1 = theta1

        if theta2 == None:
            self.theta2 = np.random.rand(hidden_size + 1, output_size) - 0.5
        else:
            self.theta2 = theta2

    def stoc_gradient(self, X, y, grad_func=mse_grad):
        # Compute the gradient with input cost function for one element 
        curr_example = X
        curr_example = np.insert(curr_example, 0, 1.)
        sigmoid = np.vectorize(self.sigmoid)
        vec_tanh = np.vectorize(tanh)
        z2 = curr_example.dot(self.theta1)
        a2 = vec_tanh(z2)
        a2 = np.insert(a2, 0, 1.)
        z3 = a2.dot(self.theta2)
        a3 = sigmoid(z3)

        delta_final = grad_func(a3, y) * self.sigmoid_grad(z3)
        delta_hidden = self.theta2.dot(delta_final)[1:] * self.tanh_grad(z2)

        # Output gradient vectors
        grad_input = np.outer(curr_example, delta_hidden)
        grad_hidden = np.outer(a2, delta_final)
        return (grad_input, grad_hidden)

    def stoc_gradient_cross(self, X, y, grad_func=mse_grad):
        curr_example = X
        curr_example = np.insert(curr_example, 0, 1.)
        sigmoid = np.vectorize(self.sigmoid)
        vec_tanh = np.vectorize(tanh)
        z2 = curr_example.dot(self.theta1)
        a2 = vec_tanh(z2)
        a2 = np.insert(a2, 0, 1.)
        z3 = a2.dot(self.theta2)
        a3 = sigmoid(z3)

        delta_final = grad_func(a3, y)
        delta_hidden = self.theta2.dot(delta_final)[1:] * self.tanh_grad(z2)

        grad_input = np.outer(curr_example, delta_hidden)
        grad_hidden = np.outer(a2, delta_final)

        input_reg = lmbda * np.vstack( (np.zeros( (1, self.theta1.shape[1]) ), self.theta1[1:]) )
        hidden_reg = lmbda * np.vstack( (np.zeros( (1, self.theta2.shape[1]) ), self.theta2[1:]) )
        grad_input += input_reg
        grad_hidden += hidden_reg
        return (grad_input, grad_hidden)

    def predict(self, input_X):
        sigmoid = vectorize(self.sigmoid)
        vec_tanh = vectorize(tanh)
        curr_example = np.hstack( (np.ones( (input_X.shape[0], 1) ), input_X) )
        z2 = curr_example.dot(self.theta1)
        # a2 = sigmoid(z2)
        a2 = vec_tanh(z2)
        a2 = np.hstack( (np.ones( (a2.shape[0], 1) ), a2) )
        z3 = a2.dot(self.theta2)
        a3 = sigmoid(z3)
        return a3

    def predict_one(self, input_X):
        sigmoid = vectorize(self.sigmoid)
        vec_tanh = vectorize(tanh)
        curr_example = np.insert(input_X, 0, 1)
        z2 = curr_example.dot(self.theta1)
        # a2 = sigmoid(z2)
        a2 = vec_tanh(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = a2.dot(self.theta2)
        a3 = sigmoid(z3)
        return a3

    def cost(self, pred_y, y):
        diff = y - pred_y
        diff *= diff
        return 0.5*np.sum(diff)/float(len(diff))

    def cross_cost(self, pred_y, y):
        vec_log = vectorize(self.log_mod)
        diff = y*vec_log(pred_y) + (1 - y)*vec_log(1 - pred_y)
        return -np.sum(diff)/float(len(diff))

    def log_mod(self, x):
        if x <= 0:
            return -40.
        return log(x)

    def class_error(self, pred_y, y):
        # Compute the current classification error
        pred = []
        actual = []
        for i in range(pred_y.shape[0]):
            curr_pred, curr_y = pred_y[i], y[i]
            pred.append(max([(curr_pred[i], i) for i in range(2)])[1])
            actual.append(max([(curr_y[i], i) for i in range(2)])[1])
        pred, actual = np.array(pred), np.array(actual)
        return (len(pred) - np.count_nonzero(pred - actual))/float(len(pred))

    def train(self, X, y, num_iter, grad_func=mse_grad, test=None, test_y=None):
        # Train the neural network for num_iter using stochastic gradient descent
        index = 0
        costs = []
        while index < num_iter:
            row = randint(0, X.shape[0] - 1)
            curr_X, curr_y = X[row], y[row]
            grad_input, grad_hidden = self.stoc_gradient(curr_X, curr_y, grad_func)
            self.theta1 -= learning_rate*grad_input
            self.theta2 -= learning_rate*grad_hidden
            if (index + 1) % 100 == 0:
                results = self.predict(X)
                print(index)
                print(self.cost(results, y))
                print(self.cross_cost(results, y))
                # print(self.class_error(results, y))
                if test != None:
                    print(self.cost(self.predict(test), test_y))
                    print(self.cross_cost(self.predict(test), test_y))
                    print(self.class_error(self.predict(test), test_y))
                np.save('weights/theta1.npy', self.theta1)
                np.save('weights/theta2.npy', self.theta2)
            index += 1

    def sigmoid(self, x):
        if x < -709:
            return 0.
        return 1/(1 + exp(-x))

    def sigmoid_grad(self, x):
        # Sigmoid gradient activation for output layer
        val = np.vectorize(self.sigmoid)(x)
        return val*(1 - val)

    def tanh_grad(self, x):
        # tanh activation function for hidden layers
        val = np.vectorize(tanh)(x)
        return 1 - val**2

    def grad_check(self, X, y):
        # Compute the analytical gradient to check gradient computation
        epsilon = 1e-6
        check_grad = np.zeros(self.theta2.shape)
        prev_theta2 = self.theta2
        self.theta2 = self.theta2[:]
        for i in range(0, self.theta2.shape[0]):
            for j in range(0, self.theta2.shape[1]):
                self.theta2[i, j] += epsilon
                pred_y = self.predict_one(X)
                plus_grad = self.cost(pred_y, y)
                self.theta2[i, j] -= 2*epsilon
                pred_y = self.predict_one(X)
                neg_grad = self.cost(pred_y, y)
                check_grad[i, j] = (plus_grad - neg_grad)/(2*epsilon)
                self.theta2[i, j] += epsilon
        self.theta2 = prev_theta2
        return check_grad

