from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import time
from VisualizationUtils import VisualizationUtils
from With_Testing.DataUtilsWithTesting import DataUtils
import numpy as np

#Variables
dataUtils = DataUtils()
graph = VisualizationUtils()
index_train = []

#Load dataset
x_train, y_train = load_svmlight_file("svmguide1.txt")

dimensions = np.arange(10, x_train.shape[0],100)

def generate_indices():
    for i in range(x_train.shape[0]):
        index_train.append(i)

for i in range(len(dimensions)):

   #Creation of the indices for subsets
    if (i != len(dimensions) - 1):
        dataUtils.select_random_index(dimensions[i], index_train, x_train)
    else:
        dataUtils.generate_indices(index_train, x_train)
        generate_indices()

    startTime = time.time()
    # Creation of subsets
    subset_xtrain = x_train.tocsr()[index_train, :]
    subset_ytrain = y_train[index_train]

    # Regression model
    logreg_model = LogisticRegression(C=2.0)

    # Train the model
    logreg_model.fit(subset_xtrain, subset_ytrain)

    # Predictions
    pred = logreg_model.predict(subset_xtrain)

    # Updating variables
    run_time = time.time() - startTime
    dataUtils.update_logistic_error(subset_ytrain,pred)
    dataUtils.update_variables(i, dimensions,logreg_model, run_time)
    index_train = []

#Plot Error and Samples

graph.plot_logistic_error(dataUtils.data_size, dataUtils.error)

#Plot CPU time and Samples
graph.plot_logistic_cpu(dataUtils.data_size,dataUtils.cpu_time)

print(logreg_model.coef_)
print(dataUtils.regr_coef[10])
print(dataUtils.regr_coef[10][0][0])
print(dataUtils.regr_coef[10][0][1])
print(dataUtils.regr_coef[10][0][2])
print(dataUtils.regr_coef[10][0][3])

#Plot Regression coefficients and Samples for each class of the data
graph.plot_logistic_coef(graph.get_features_logistic(10, dataUtils.regr_coef))

lastIndex = len(dimensions)-1
graph.plot_logistic_coef(graph.get_features_logistic(lastIndex, dataUtils.regr_coef))

