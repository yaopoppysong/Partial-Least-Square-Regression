#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
# Test Code of the function       #
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# load package
import numpy as np
import numpy.random as npr
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression

# Set initial data set
npr.seed(123)
X = np.random.normal(0, 1, (10, 3))
Y = np.random.normal(0, 1, (10, 2))
# normalize the matrices X and Y
X = scale(X)
Y = scale(Y)

# obtain the prediction values from the Partial Least Square
Y_pred = pls_prediction(X, Y, X, 3, 1e-06)
Y_pred

# Calculate the Prediction Residual Sum of Squares
np.sum((Y-Y_pred)**2)

# Check the Results with Package in Python and R
pls1 = PLSRegression(n_components = 3)
pls1.fit(X, Y)
Y_pred1 = pls1.predict(X)
print(Y_pred1)
np.sum((Y-Y_pred1)**2)

np.testing.assert_almost_equal(Y_pred, Y_pred1, decimal = 2) # Ture when the error bound is in 0.01

# Check the Number of Components
pls_ncomponents(X, Y, 1e-06)
