import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import tree

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

prediction = regressor.predict([[6.8]])
print("Predicted Salary for Level 6.8 is", prediction)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red", label="Actual")
plt.plot(x_grid, regressor.predict(x_grid), color="blue", label="Predicted")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

export_graphviz(regressor, out_file="name.dot", feature_names=["Experience"])

tree.plot_tree(regressor)
plt.show()
