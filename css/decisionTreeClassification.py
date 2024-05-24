import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Social.csv')
print(df)
X = df.iloc[:, 2:4]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,Y_train)

from sklearn import tree
tree.plot_tree(classifier)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy: ", accuracy_score(Y_test, y_pred))

new_data_point = [[26, 35050]]
predicted_class = classifier.predict(new_data_point)
print("Predicted class for ", new_data_point, "is: ", predicted_class)

from sklearn.tree import export_graphviz
export_graphviz(classifier, out_file='classtree.dot', feature_names=X.columns, class_names=['Not Purchased', 'Purchased'])