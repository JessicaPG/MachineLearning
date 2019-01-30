from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import time
from VisualizationUtils import VisualizationUtils
from DataUtilsWithTesting import DataUtils
from sklearn.metrics import confusion_matrix

#Variables
dataUtils = DataUtils()
graph = VisualizationUtils()
index_train = []

#Load dataset
x, y = load_svmlight_file("wineDataset.txt")

#Split dataset
x_train,x_test,y_train,y_test = dataUtils.split_dataset(x,y)
dimensions = [4,10,20,30,40,50,60,70,80,90,100,120,130,x_train.shape[0]]

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
    logreg_model = LogisticRegression()

    # Train the model
    logreg_model.fit(subset_xtrain, subset_ytrain)

    # Predictions
    pred = logreg_model.predict(x_test)

    # Updating variables
    run_time = time.time() - startTime
    dataUtils.update_logistic_error(y_test,pred)
    dataUtils.update_variables(i, dimensions,logreg_model, run_time)
    index_train = []

#Plot Error and Samples

graph.plot_logistic_error(dataUtils.data_size, dataUtils.error)

#Plot CPU time and Samples
graph.plot_logistic_cpu(dataUtils.data_size,dataUtils.cpu_time)

# Plot Regression coefficients and Samples for each class of the data
graph.plot_logistic_coef(graph.get_features_logistic(2,dataUtils.regr_coef,0))
graph.plot_logistic_coef(graph.get_features_logistic(2,dataUtils.regr_coef,1))
graph.plot_logistic_coef(graph.get_features_logistic(2,dataUtils.regr_coef,2))

graph.plot_logistic_coef(graph.get_features_logistic(13,dataUtils.regr_coef,0))
graph.plot_logistic_coef(graph.get_features_logistic(13,dataUtils.regr_coef,1))
graph.plot_logistic_coef(graph.get_features_logistic(13,dataUtils.regr_coef,2))

