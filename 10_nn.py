import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) # Iris Setosa?
per_clf = Perceptron()
per_clf.fit(X, y)
y_pred1 = per_clf.predict([[2, 0.4]])
y_pred0 = per_clf.predict([[2, 0.5]])
print(y_pred1, y_pred0)
