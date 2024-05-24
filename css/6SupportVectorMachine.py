import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

# Importing the dataset
df = pd.read_csv("cancer.csv")
df.replace("?", -99999, inplace=True)
df.drop('id', inplace=True, axis=1)
x = np.array(df.drop(["classes"], axis=1))
y = np.array(df["classes"])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

# Training the Support Vector Machine model on the Training set
classifier = svm.SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
confidence = classifier.score(X_test, y_test)
print("Confidence:", confidence)

# Making a single prediction
testCase = [5, 1, 2, 9, 8, 3, 5, 7, 2]
print("prediction:", classifier.predict([testCase]))