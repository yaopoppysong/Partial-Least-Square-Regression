# Partial-Least-Square-Regression

Team Members: 
Yao Song, 
Yi Zhao

Functions
1. Function "normal", standardize the data set to be 0 mean and 1 standard deviation
2. Function "f1", calculate vector w and t, be used in the loop of the main function "pls_fit"
3. Function "pls_fit", use Partial Least Square Regression Algorithm to obtain necessary matrices 
   with specific number of components and convergence bound
4. Function "pls_prediction", provide prediction values under specific number of components and convergence bound
5. Function "pls_ncomponents", provide the prediction residual sum of square for each number of components to obtain optimal value
 
Parameters
1. X is the matrix of independent variables
2. Y is the matrix of dependent variables
3. tol is the convergence bound
4. n_comp is the number of component
