# load package
import numpy as np
import numpy.random as npr
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# PLS package
class PLS:
    "Partial Least Square Regression Class"
    
    def __init__(self, X, Y, X_test,n_comp, tol):
      """Function to initialize the parameters
        X is the matrix of independent variables
        Y is the matrix of dependent variables
        X_test is the matrix of dependent variables in testing data set
        tol is the convergence bound
        n_comp is the number of component"""
      
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.n_comp = n_comp
        self.tol = tol
    
    def pls_fit(self, n_comp):
        """Function to get the Partial Least Square Regression Coefficient
        n_comp is the number of component"""
        
        from numpy import linalg as LA
    
        # get variable numbers of X and Y
        n = self.X.shape[0]
        m = self.X.shape[1]
        k = self.Y.shape[1]

    
        # normalize the matrices X and Y
        X = scale(self.X)
        Y = scale(self.Y)
    
        # set initial value for the scores of Y and X
        E = X
        F = Y
        u = Y[:, 0][:, None]
        t = X[:, 0][:, None]
        W = np.zeros((m, n_comp))
        Q = np.zeros((k, n_comp))
        P = np.zeros((m, n_comp))
        B = np.zeros(n_comp)
        U = np.zeros((n, n_comp))
        T = np.zeros((n, n_comp))
    
        def f1(u, E):
            "Function to calculate w and t"
            from numpy import linalg as LA
            w = E.T @ u / (u.T @ u)
            w = w/LA.norm(w)
            t = E @ w /(w.T @ w)
            return (w, t)
    
        # The PLS algorithm
        # check the residuals of E and F
        for i in range(n_comp):
            # check the convergency of t (the scores of X)
            if k == 1:
                w = f1(Y, E)[0]
                t = f1(Y, E)[1]
            else: 
                while np.allclose(t, f1(u, E)[1], atol = self.tol, rtol = self.tol) == False:
                    w = f1(u, E)[0]
                    t = f1(u, E)[1]
                    q = F.T @ t/(t.T @ t)
                    q = q/LA.norm(q)
                    u = F @ q /(q.T @ q)
                
            p = E.T @ t/(t.T @ t)
            t = t * LA.norm(p)        
            w = w * LA.norm(p)
            p = p/LA.norm(p)
            b = u.T @ t/(t.T @ t)
        
            E = E - t @ p.T
            if k == 1:
                F = F - b * t
            else:
                F = F - b * t @ q.T
                Q[:, i][:, None] = q
            
            W[:, i][:, None] = w     
            P[:, i][:, None] = p
            B[i] = b
            U[:, i][:, None] = u 
            T[:, i][:, None] = t 
    
        if k == 1:
            Q = np.ones((1, n_comp))
        
        return m, k, B, W, Q, P 
    
    def pls_prediction(self, X_test, n_comp):
        """Function to get the Partial Least Square Regression Prediction
        X_test is the matrix of independent variables in testing data set
        n_comp is the number of component"""
    
        # normalize the matrices X and Y
        X_test = scale(X_test)
        
        # get variable numbers of X and Y
        n = X_test.shape[0]
        
        # recall pls_fit function
        m, k, B, W, Q, P = self.pls_fit(n_comp)
    
        # The prediction
        E = X_test
        T_hat = np.zeros((n, n_comp))
        Y_pred = np.zeros((n, k))
    
        if n_comp > m:
            print("Error: number of components is larger than number of X variables")
        else:
            for i in range(n_comp):
                T_hat[:, i] = E @ W[:, i]
                E = E - T_hat[:, i][:, None] @ P[:, i][None, :]
                if k == 1:
                    Y_pred += B[i] * T_hat[:, i][:, None]
                else:
                    Y_pred += B[i] * T_hat[:, i][:, None] @ Q[:, i][None, :]
    
        return Y_pred
    
    def pls_ncomponents(self):
        """Function to get the Partial Least Square Regression Prediction"""
        
        # get variable numbers of X and Y
        m = self.X.shape[1]
    
        # create new variables
        PRESS = np.zeros(m)
    
        for i in range(m):
            Y_pred = self.pls_prediction(X, i+1)
            PRESS[i] = np.sum((self.Y - Y_pred)**2)
    
        return PRESS, plt.plot(np.arange(1, m+1, 1), PRESS)
