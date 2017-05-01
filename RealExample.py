# Using Real Data Set

# load package
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# read in data set
wine = pd.read_excel('wine.xlsx')
wine.head()
wine_new = wine.copy()

# normalize the data set
n = len(wine.columns)
for i in range(n):
    wine_new.ix[:, i] = scale(wine.ix[:, i])
wine_new = np.array(np.matrix(wine_new))

# separate the data set
wine_newX = wine_new[:, 3:]
wine_newY = wine_new[:, :3]
wine_newX

# Using partial least square function
fit = PLS(wine_newX, wine_newY, wine_newX, 3, 1e-06)
Y_pred = fit.pls_prediction(wine_newX, 3)
Y_pred = pls_prediction(wine_newX, wine_newY, wine_newX, 3, 1e-06)
np.sum((wine_newY-Y_pred)**2) # PRESS

# Using Partial Least Square Package in Python
pls1 = PLSRegression(n_components = 3)
pls1.fit(wine_newX, wine_newY)
Y_pred1 = pls1.predict(wine_newX)
np.sum((wine_newY-Y_pred1)**2) # PRESS

# Check the number of components by PRESS
fit.pls_ncomponents()
