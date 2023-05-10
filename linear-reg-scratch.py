import numpy as np

class Linear_Regression():
    
    #initiating parameters
    def __init__(self, learning_rate, no_of_iterations):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    
    def fit(self, x, y):
        
        #no of trainig examples(m) and no of features(n)
        self.m, self.n = x.shape

    
    def update_weight(self):
        pass
    
    def predict(self):
        pass
    