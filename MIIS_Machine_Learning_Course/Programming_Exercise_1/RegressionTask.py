from sklearn import linear_model
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
x_train, y_train = load_svmlight_file("dataset.txt")

dimensions = np.arange(50, x_train.shape[0],150)

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
    regr_model = linear_model.LinearRegression()

    # Train the model
    regr_model.fit(subset_xtrain, subset_ytrain)

    # Predictions
    pred = regr_model.predict(subset_xtrain)

    # Updating variables
    run_time = time.time() - startTime
    dataUtils.update_linear_error(subset_ytrain, pred)
    dataUtils.update_variables(i, dimensions,regr_model, run_time)
    index_train = []

#Plot Error and Samples
graph.plot_linear_error(dataUtils.data_size, dataUtils.error)

#Plot CPU time and Samples
graph.plot_linear_cpu(dataUtils.data_size,dataUtils.cpu_time)

# Plot Regression coefficients and Samples
graph.plot_linear_coef(graph.get_features_linear(0,dataUtils.regr_coef))
graph.plot_linear_coef(graph.get_features_linear(15,dataUtils.regr_coef))

