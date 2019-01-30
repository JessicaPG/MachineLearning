from sklearn.datasets import load_svmlight_file
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import validation_curve

path = '/Users/jessie/PycharmProjects/ProgrammingEx2/dataset.txt'
alpha_range = np.logspace(-15, 15, 20)
x, y = load_svmlight_file(path)


#### Cross validation
train_scores, valid_scores = validation_curve(MLPClassifier(activation='logistic',max_iter=100), x, y, "alpha", alpha_range, scoring = 'accuracy', cv=5)

pyplot.title("Validation Curve with MLP")
pyplot.xlabel(r'$\alpha $')
pyplot.ylabel("Score: Accuracy")
pyplot.xscale('log')
pyplot.plot(alpha_range, valid_scores.mean(axis=1), label='cross-validation')
pyplot.plot(alpha_range, train_scores.mean(axis=1), label='training')
pyplot.legend(loc="best")
pyplot.show()

print(len(valid_scores))
print(len(alpha_range))

pyplot.title("Validation Curve with MLP")
pyplot.xlabel(r'$\alpha $')
pyplot.ylabel("Error")
pyplot.xscale('log')
pyplot.plot(alpha_range, 1- valid_scores.mean(axis=1), label='cross-validation',linewidth=2.0)
pyplot.plot(alpha_range, 1- train_scores.mean(axis=1), label='training',linewidth=2.0)
pyplot.legend(loc="best")
pyplot.show()


## Learning curve

# pyplot.title("Learning Curve")
# pyplot.xlabel("Size dataset")
# pyplot.ylabel("Error")
# train_sizes, train_scores, valid_scores = learning_curve(MLPClassifier(activation='logistic'), x, y, train_sizes=[50, 80, 110,300,500,1000,1500,1700,2000,2100], cv=5)
# pyplot.plot(train_sizes, 1- valid_scores.mean(axis=1), label='cross-validation',linewidth=2.0)
# pyplot.plot(train_sizes, 1- train_scores.mean(axis=1), label='training',linewidth=2.0)
# pyplot.legend(loc="best")
# pyplot.show()
#
