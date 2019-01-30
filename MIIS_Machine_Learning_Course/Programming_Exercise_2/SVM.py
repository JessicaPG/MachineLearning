from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as pyplot
from prettytable import PrettyTable
from matplotlib.colors import Normalize

path = '/Users/jessie/PycharmProjects/ProgrammingEx2/dataset.txt'
n_fold = 3

# Load data
x, y = load_svmlight_file(path)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Set parameters CV
n = 10
c_value = np.logspace(-5,8,n)
gamma_value = np.logspace(-15,5,n)

parameters = [{'kernel': ['rbf'], 'gamma': gamma_value,
                     'C': c_value}]

# Create SVM model
model = GridSearchCV(SVC(), parameters, scoring='accuracy', cv=KFold(n_fold))
model.fit(x_train, y_train)
pred = model.best_estimator_.predict(x_test)
accu = (1-(accuracy_score(y_test,pred)))



# Retrain the dataset with best C and gamma estimators
modelT = SVC(C= model.best_estimator_.C, gamma = model.best_estimator_.gamma,kernel = 'rbf')
modelT.fit(x,y)
predT = modelT.predict(x)
accuT = (1- (accuracy_score(y,predT)))



# Save variables for plot and tables
c_array = []
score_array =[]
gamma_array = []

for obj in model.cv_results_['params']:
    c_array.append(obj["C"])
    gamma_array.append(obj["gamma"])
for num in model.cv_results_['mean_test_score']:
    score_array.append(num)


# Create table with parameters and scores of the model
table = PrettyTable()
table.field_names = ["Parameters", "Score"]
table.add_row(["Best scores Train Subset",model.best_score_ ])
table.add_row(["Best C",model.best_estimator_.C ])
table.add_row(["Best Gamma", model.best_estimator_.gamma])
table.add_row(["Train subset classification error", accu])
table.add_row(["Classification error", accuT])
print(table)

# Create table of all combinations of gamma and C and its validation score
# tableScores = PrettyTable()
# tableScores.field_names = ["Param C", "Param Gamma",  "Cross-validation"]
#
# for i in range(len(c_array)):
#     for in_gamma in range(len(gamma_array)):
#         #tableScores.add_row([c_value[i], gamma_value[in_gamma], score_array[in_gamma]])
#         print(c_value[i])
        #print(gamma_value[in_gamma])
        #print( score_array[in_gamma])
#print(tableScores)


# Plot gamma against CV error for each value of C

pyplot.xscale('log')
pyplot.xlabel(r'$\gamma $')
pyplot.ylabel('CV Error')

i = 0
while i < len(gamma_array) and i < len(score_array):

    pyplot.plot(gamma_array[i:i+n], score_array[i:i+n], label='C = ' + str(c_array[i]), linewidth=2.0)
    i += n
pyplot.legend(loc="best")
pyplot.show()

## Plot headMap

# Code for plot extracted from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

scores = model.cv_results_['mean_test_score'].reshape(len(c_value),
                                                     len(gamma_value))

pyplot.figure(figsize=(8, 6))
pyplot.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
pyplot.imshow(scores, interpolation='nearest', cmap=pyplot.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
pyplot.xlabel('gamma')
pyplot.ylabel('C')
pyplot.colorbar()
pyplot.xticks(np.arange(len(gamma_value)), gamma_value, rotation=45)
pyplot.yticks(np.arange(len(c_value)), c_value)
pyplot.title('Validation accuracy')
pyplot.show()
