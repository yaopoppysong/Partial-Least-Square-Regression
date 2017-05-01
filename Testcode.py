#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
# Test Code of the function       #
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# load package
import numpy as np
import numpy.random as npr
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

# Set initial data set
npr.seed(123)
X = np.random.normal(0, 1, (10, 3))
Y = np.random.normal(0, 1, (10, 2))
# normalize the matrices X and Y
X = scale(X)
Y = scale(Y)

# obtain the prediction values from the Partial Least Square
fit = PLS(X, Y, X, 3, 1e-06)
Y_pred = fit.pls_prediction(X, 3)
Y_pred

# Calculate the Prediction Residual Sum of Squares
np.sum((Y-Y_pred)**2)

# Check the Results with Package in Python and R
pls1 = PLSRegression(n_components = 3)
pls1.fit(X, Y)
Y_pred1 = pls1.predict(X)
print(Y_pred1)
np.sum((Y-Y_pred1)**2)

# Check the Number of Components
fit.pls_ncomponents()
