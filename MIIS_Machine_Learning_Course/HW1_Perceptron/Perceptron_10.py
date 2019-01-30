import numpy as np
from matplotlib import pyplot
import time
import sys


class Perceptron(object):

    def __init__(self, n, d):
        self.data = np.random.uniform(-1, 1, size=(n, d + 1))
        ## Add a column named bias with value 1 to the dataset
        self.data[:, 0] = 1
        self.d = d

    def classification(self, w):
        # Obtain the label/class of each point
        labels = np.sign(np.dot(self.data, w))
        # Transform label data in int type
        return labels.astype(int)

    def perceptron_algorithm(self):
        w = np.random.uniform(-1, 1, size=self.d + 1)
        true_labels = self.classification(w)
        weight = np.zeros(self.d + 1)
        max_iter = 1000

        for it in range(1, max_iter):
            is_updated = False
            error = 0

            for i in range(len(self.data)):
                predicted_label = self.classification(weight)

                if predicted_label[i] != true_labels[i]:
                    error += 1
                    is_updated = True
                    weight += true_labels[i] * self.data[i]

            if not is_updated:
                break

        if is_updated:
            print("No converge %i" % it)
            print("Errors %i" % error)

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

    print('For %s numbers, %s dimensions a total of %s iterations have been necessary to converge; ' % (
    str(num), str(dim), str(iteration)))
    print('Total time %s' % (str(endTime - startTime)))

