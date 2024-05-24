import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating the model
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", ac)
print("Confusion Matrix:\n", cm)

# Making a single prediction and plotting boxplots
arr_for_pred = np.array([70, 760000]).reshape(1, -1)
value_pred = sc.transform(arr_for_pred)
p1 = classifier.predict(value_pred)
print("Prediction:", p1)

# Plotting boxplots
plt.boxplot(X_train)
plt.title('Boxplot of Scaled Features (X_train)')
plt.show()

plt.boxplot(X_test)
plt.title('Boxplot of Scaled Features (X_test)')
plt.show()
