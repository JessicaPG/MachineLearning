import matplotlib.pyplot as pyplot

class VisualizationUtils:
    """Class VisualizationUtils contains all the functions related with the visualization of the data"""

    def plot_linear_error(self, data_size, error):
        pyplot.plot(data_size,error, c='indigo',linewidth=2.0)
        pyplot.title("Square Loss as a function of data samples")
        pyplot.xlabel("Data Samples")
        pyplot.ylabel("Square loss error")
        pyplot.grid(linestyle='-', linewidth=0.3)
        pyplot.ylim(0,6)
        pyplot.show()

    def plot_logistic_error(self,data_size,error):
        pyplot.plot(data_size,error,'ro', c='orange')
        pyplot.plot(data_size,error, c='indigo',linewidth=2.0)
        pyplot.title("Accuracy metric as a function of data samples")
        pyplot.xlabel("Data Samples")
        pyplot.ylabel("Accuracy metric")
        pyplot.ylim(0.0, 1.5)
        pyplot.grid(linestyle='-', linewidth=0.3)
        pyplot.show()


    def plot_linear_cpu(self, data_size,cpu_time):
        pyplot.plot(data_size, cpu_time, c='indigo',linewidth=2.0)
        pyplot.title("CPU time as a function of data samples")
        pyplot.xlabel("Data Samples")
        pyplot.ylabel("CPU time")
        pyplot.grid(linestyle='-', linewidth=0.3)
        pyplot.show()


    def plot_logistic_cpu(self, data_size,cpu_time):
        pyplot.plot(data_size, cpu_time,'ro', c='pink')
        pyplot.plot(data_size, cpu_time, c='indigo',linewidth=2.0)
        pyplot.title("CPU time as a function of data samples")
        pyplot.xlabel("Data Samples")
        pyplot.ylabel("CPU time")
        pyplot.grid(linestyle='-', linewidth=0.3)
        pyplot.show()

    def plot_linear_coef(self, features):
        pyplot.bar(range(len(features)), features.values(), align='center', color = 'lavender')
        pyplot.xticks(range(len(features)), features.keys(), rotation='vertical')
        pyplot.axhline(y=0, linestyle='--', color='black', linewidth=3)
        pyplot.title("Regression coefficients")
        pyplot.xlabel("Features")
        pyplot.ylabel("Regr. coef")
        pyplot.show()


    def plot_logistic_coef(self,features):
        pyplot.xticks(range(len(features)), features.keys(), rotation='vertical')
        pyplot.bar(range(len(features)), features.values(), align='center', color = 'lavender')
        pyplot.axhline(y=0, linestyle='--', color='black', linewidth=3)
        pyplot.title("Regression coefficients")
        pyplot.xlabel("Features")
        pyplot.ylabel("Regr. coef")
        pyplot.show()


    def get_features_linear(self, dim, regr_coef):
        return {'Sex': regr_coef[dim][0], 'Length': regr_coef[dim][1], 'Diameter': regr_coef[dim][2],
                    'Height': regr_coef[dim][3], 'Whole weight': regr_coef[dim][4],
                    'Whole weight': regr_coef[dim][5], 'Viscera weight': regr_coef[dim][6], 'Shell weight': regr_coef[dim][7]}

    def get_features_logistic(self,dim,regr_coef):
        return {'Att1': regr_coef[dim][0][0], 'Att2': regr_coef[dim][0][1], 'Att3': regr_coef[dim][0][2], 'Att4': regr_coef[dim][0][3]}

