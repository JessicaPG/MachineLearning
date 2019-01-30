
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

class DataUtils:
    """Class DataUtils contains all the functions related with the generation or manipulation of the data"""

    def __init__(self):
        self.data_size = []
        self.error = []
        self.cpu_time = []
        self.regr_coef = []


    def split_dataset(self, x, y):
        """Creation of train and test subsets"""
        #Calculate 20% of samples from the dataset
        num = int(round(x.shape[0] * 0.2))
        x_train = x[:-num]
        x_test = x[-num:]
        y_train = y[:-num]
        y_test = y[-num:]
        return x_train,x_test,y_train,y_test

    def select_random_index(self, dim,index_train,x_train):
        """Generate random index based on the given dimension except for when the whole training dataset is taken"""
        while len(index_train) < dim:
            num = random.randint(0, x_train.shape[0] - 1)
            if num not in index_train:
                index_train.append(num)


    def generate_indices(self, index_train,x_train):
        """Generate an index array with length equal to x_train variable"""
        for i in range(x_train.shape[0]):
            index_train.append(i)


    def update_linear_error(self,y_test, pred):
        self.error.append(mean_squared_error(y_test, pred))

    def update_logistic_error(self, y_test, pred):
        self.error.append(accuracy_score(y_test, pred))

    def update_variables(self,i, dimensions,model, run_time):
        self.data_size.append(dimensions[i])
        self.cpu_time.append(run_time)
        self. regr_coef.append(model.coef_)

