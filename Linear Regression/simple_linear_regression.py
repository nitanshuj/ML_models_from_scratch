"""
Simple Linear Regression
========================
This is the code for Linear Regression with 
> 1 Predictor/Independent variable 
> 1 Target Variable
"""


import numpy as np, pandas as pd

class Simple_LR_1:
    def __init__(self):
        self.b1 = None
        self.b0 = None
    
    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        numer_b1 = np.sum(np.subtract(X, X_mean)*np.subtract(y, y_mean))
        denom_b1 = np.sum(np.square(np.subtract(X, X_mean)))
        self.b1 = numer_b1/denom_b1
        self.b0 = y_mean - (self.b1*X_mean)

    def predict(self, X):
        return self.b0+(self.b1*X)




if __name__ == "__main__":
    print("Simple Linear Regression")
    X = [1,2,3,4,5,6,7,8,9,10,11]
    y = [1,4,6,8,10,12,14,16,18,20,22]
    slr = Simple_LR_1()
    slr.fit(X,y)    
    print(slr.predict(25))

