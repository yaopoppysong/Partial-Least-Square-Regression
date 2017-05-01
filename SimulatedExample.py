# Using Simulated Data Set

# load package
import numpy as np
import numpy.random as npr
from sklearn.preprocessing import scale

# Set initial data set
npr.seed(123)
X = np.random.normal(0, 1, (100, 6))
Y = np.random.normal(0, 1, (100, 2))

# normalize the matrices X and Y
X = scale(X)
Y = scale(Y)

# Separate the data set
X_train = X[:70, :]
X_test = X[70:, :]
Y_train = Y[:70, :]
Y_test = Y[70:, :]

# Using Partial Least Square Prediction Function
Y_pred = pls_prediction(X_train, Y_train, X_test, 3, 1e-06)
np.sum((Y_test-Y_pred)**2) # PRESS

# Using Partial Least Square Package in Python
pls1 = PLSRegression(n_components = 3)
pls1.fit(X_train, Y_train)
Y_pred1 = pls1.predict(X_test)
np.sum((Y_test-Y_pred1)**2) # PRESS
