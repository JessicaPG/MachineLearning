import numpy as np
from matplotlib import pyplot
import time
import sys

class Perceptron(object):

    def __init__(self, n, d):
        self.points = np.random.uniform(-1, 1, size=(n,d))
        self.data = np.random.uniform(-1, 1, size=(n, d))
        self.label = np.zeros(n).astype(int)


    def classification(self):
        X1 = self.points[0][0]
        X2 = self.points[1][0]
        Y1 = self.points[0][1]
        Y2 = self.points[1][1]

        slope = (Y2 - Y1) / (X2 - X1)
        b = Y1 - (slope * X1)
        linspace = np.linspace(-1, 1)

        for i in range(len(self.data)):
            x = self.data[i][0]
            y = self.data[i][1]

            value = (X2 - X1) * (y - Y1) - (x - X1) * (Y2 - Y1)
            if value > 0:
                self.label[i] = 1
            else:
                self.label[i] = -1

        self.plot_line(slope, linspace, b, 'orange','Target function')


    def plot(self, bias, weight):

        for i in range(len(self.data)):
            if self.label[i] == 1:
                pyplot.scatter(self.data[i][0], self.data[i][1], c='green', marker='+')
            else:
                pyplot.scatter(self.data[i][0], self.data[i][1], c='blue', marker='_')

        # Calculate line slope and intercept
        slope = -(weight[0]) / (weight[1])
        intercept = -bias / weight[1]
        linspace = np.linspace(-1, 1)

        pyplot.title('Perceptron Learning Algorithm')
        self.plot_line(slope, linspace, intercept, 'red', 'Hypothesis')
        pyplot.legend()
        pyplot.show()



    def plot_line(self, slope, linspace, b,color,label):
        pyplot.xlim(-1, 1)
        pyplot.ylim(-1, 1)
        pyplot.plot(linspace, slope * linspace + b, c=color, label= label)


    def perceptron_algorithm(self):
        self.classification()

        weight = np.zeros(len(self.data[0]))
        max_iter = 1000
        bias = 0
        error = 0

        for it in range(1, max_iter):
            is_updated = False
            error = 0
            for i in range(0, len(self.data)):
                activation = bias
                activation += weight.T.dot(self.data[i])

                if int(np.sign(activation * self.label[i])) != 1:
                    error += 1
                    is_updated = True
                    weight += self.label[i] * self.data[i]
                    bias += self.label[i]

            if not is_updated:
                break

        if is_updated:
            print("No converge %i" % it)
            print("Errors %i" % error)

        self.plot(bias, weight)
        return it


if __name__ == "__main__":

    try:
        num = int(sys.argv[1])
        dim = int(sys.argv[2])
    except:
        "This operation needs 2 arguments: numbers and dimension"

    startTime = time.time()
    perceptron = Perceptron(num, dim)
    iteration = perceptron.perceptron_algorithm()
    endTime = time.time()

    print('For %s numbers, %s dimensions a total of %s iterations have been necessary to converge; ' % (str(num), str(dim), str(iteration)))
    print('Total time %s' %(str(endTime - startTime)))






