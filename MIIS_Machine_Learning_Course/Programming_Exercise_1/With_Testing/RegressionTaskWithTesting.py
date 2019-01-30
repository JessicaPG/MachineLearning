from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
import time
from VisualizationUtils import VisualizationUtils
from DataUtilsWithTesting import DataUtils

#Variables
dataUtils = DataUtils()
graph = VisualizationUtils()
index_train = []

#Load dataset
x, y = load_svmlight_file("dataset.txt")

#Split dataset
x_train,x_test,y_train,y_test = dataUtils.split_dataset(x,y)
dimensions = [4,40,100,200,400,600,1000,1200,1600,2000,2200,2600,3000,3200,x_train.shape[0]]

def generate_indices():
    for i in range(x_train.shape[0]):
        index_train.append(i)

for i in range(len(dimensions)):
    startTime = time.time()

   #Creation of the indices for subsets
    if (i != len(dimensions) - 1):
        dataUtils.select_random_index(dimensions[i], index_train, x_train)
    else:
        dataUtils.generate_indices(index_train, x_train)
        generate_indices()

    # Creation of subsets
    subset_xtrain = x_train.tocsr()[index_train, :]
    subset_ytrain = y_train[index_train]

    # Regression model
    regr_model = linear_model.LinearRegression()

    # Train the model
    regr_model.fit(subset_xtrain, subset_ytrain)

    # Predictions
    pred = regr_model.predict(x_test)

    # Updating variables
    run_time = time.time() - startTime
    dataUtils.update_linear_error(y_test, pred)
    dataUtils.update_variables(i, dimensions,regr_model, run_time)
    index_train = []

#Plot Error and Samples
graph.plot_linear_error(dataUtils.data_size, dataUtils.error)

#Plot CPU time and Samples
graph.plot_linear_cpu(dataUtils.data_size,dataUtils.cpu_time)

# Plot Regression coefficients and Samples
graph.plot_linear_coef(graph.get_features_linear(0,dataUtils.regr_coef))
graph.plot_linear_coef(graph.get_features_linear(7,dataUtils.regr_coef))

