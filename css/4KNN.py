import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("iris.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
classifier.fit(x_train, y_train)

# predict the test results
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ')
print(cm)

print('Accuracy Metrics: ')
print(classification_report(y_test, y_pred))
print('Correct Predictions: ', accuracy_score(y_test, y_pred))
print('Incorrect Predictions: ', 1 - accuracy_score(y_test, y_pred))

# Predicting a new result
col1 = dataset.iloc[:, 0].values
col2 = dataset.iloc[:, 1].values
col3 = dataset.iloc[:, 2].values
col4 = dataset.iloc[:, 3].values
import random
yp = []
yp.append(round(random.uniform(min(col1), max(col1)), 1))
yp.append(round(random.uniform(min(col2), max(col2)), 1))
yp.append(round(random.uniform(min(col3), max(col3)), 1))
yp.append(round(random.uniform(min(col4), max(col4)), 1))
print('Randomly Generated Data: ', yp)
for_pred = np.array(yp).reshape(1, -1)
print('Predicted Species: ', classifier.predict(for_pred))  