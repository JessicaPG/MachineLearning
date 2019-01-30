from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import os


MAX_DEPTH = 5
acc = []
dim = []
crossV = []
yTrue =[]
yPred = []
exp = 10
path = '/Users/jessie/PycharmProjects/ProgrammingEx2/dataset.txt'

for i in range(exp):

    #data
    x, y = load_svmlight_file(path)
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)

    #Train the model
    tree_m = DecisionTreeClassifier(max_depth=MAX_DEPTH)
    tree_m = tree_m.fit(x_train,y_train)
    pred = tree_m.predict(x_test)


    # Update variables
    acc.append(accuracy_score(y_test,pred))
    dim.append(i)
    crossV.append(cross_val_score(estimator=tree_m, X=x, y=y, cv=5))
    yPred.append(pred)
    yTrue.append(y_test)

    # Plot the tree
    bash_command = 'dot tree' + str(i)+ '.dot -Tpng -o im'+str(i)+'.png'
    tree_plot = tree.export_graphviz(tree_m, feature_names= ['att1','att2','att3','att4'], class_names=['0','1'], filled='true', out_file= 'tree'+ str(i)+'.dot')
    os.system(bash_command)

# Print scores
table = PrettyTable()
table.field_names = ["Tree", "Accuracy", "Classification Error",  "Confusion Matrix", "Cross - Validation Error"]

for index in range(len(dim)):
    table.add_row(["Tree " + str(dim[index]), acc[index], 1-acc[index], confusion_matrix(yTrue[i],yPred[i]), crossV[index]])

print(table)



