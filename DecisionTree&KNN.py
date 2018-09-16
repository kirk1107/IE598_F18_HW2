from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt

#from pydotplus import graph_from_dot_data
#from sklearn.tree import export_graphviz

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
#print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16, stratify=y)

#print('Labels counts in y:', np.bincount(y))
#print('Labels counts in y_train:', np.bincount(y_train))
#print('Labels counts in y_test:', np.bincount(y_test))
##############################################################################
#Decision Tree

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=17)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, clf=tree, X_highlight=X_test)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()


#dot_data = export_graphviz(tree,filled=True,rounded=True, class_names=['Setosa','Versicolor','Virginica'],feature_names=['petal length', 'petal width'],out_file=None) 
#graph = graph_from_dot_data(dot_data) 
#graph.write_png('tree.png')

###############################################################################

#KNN

#Standardize 
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Method 1
k_range = np.arange(1, 26)
test_scores = []
train_scores =[]

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    y_pred_test = knn.predict(X_test_std)
    y_pred_train = knn.predict(X_train_std)
    test_scores.append(metrics.accuracy_score(y_test, y_pred_test))
    train_scores.append(metrics.accuracy_score(y_train, y_pred_train))
    
#    plt.title('#Neighbor =' +str(k)+ '\ntesting accuracy = '+str(metrics.accuracy_score(y_test, y_pred_test)))
#    plot_decision_regions(X_test_std, y_test,clf=knn)
#    plt.xlabel('petal length [standardized]')
#    plt.ylabel('petal width [standardized]')
#    plt.legend(loc='upper left')
#    plt.show()

#Plot accuracy result with different number of neighbors 
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_scores, label = 'testing accuracy')
plt.plot(k_range, train_scores, label = 'training accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#From this graph, I think k should be 5


#Method 2 (learn from the exercise in DataCamp)
#neighbors = np.arange(1, 26)
#train_accuracy = np.empty(len(neighbors))
#test_accuracy = np.empty(len(neighbors))
#
## Loop over different values of k
#for i, k in enumerate(neighbors):
#    # Setup a k-NN Classifier with k neighbors: knn
#    knn = KNeighborsClassifier(n_neighbors=k)
#
#    # Fit the classifier to the training data
#    knn.fit(X_train_std, y_train)
#    
#    #Compute accuracy on the training set
#    train_accuracy[i] = knn.score(X_train_std, y_train)
#
#    #Compute accuracy on the testing set
#    test_accuracy[i] = knn.score(X_test_std, y_test)
#
#    #Plot
#    plt.title('#Neighbors = '+ str(k)+'\n Training accuracy ='+str(train_accuracy[i])+'\nTesting accuracy ='+str(test_accuracy[i]))
#    plot_decision_regions(X_test_std, y_test, clf=knn)
#    plt.xlabel('petal length [standardized]')
#    plt.ylabel('petal width [standardized]')
#    plt.legend(loc='upper left')
#    plt.show()
#
## Generate plot
#plt.title('k-NN: Varying Number of Neighbors')
#plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
#plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
#plt.legend()
#plt.xlabel('Number of Neighbors')
#plt.ylabel('Accuracy')
#plt.show()
#    
print("My name is Jianhao Cui")
print("My NetID is: jianhao3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")




