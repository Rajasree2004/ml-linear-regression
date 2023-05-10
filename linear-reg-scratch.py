import numpy as np

class Linear_Regression():
    
    #initiating parameters
    def __init__(self, learning_rate, no_of_iterations):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    
    def fit(self, x, y):
        
        #no of trainig examples(m) and no of features(n)
        self.m, self.n = x.shape
        #initiate weight and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.x =x
        self.y =y


        #implement gradient descent

        for i in range(self.no_of_iterations):
            self.update_weight()




    def update_weight(self):
        Y_prediction = self.predict(self.x)

        #calculate gradient

        dw = (-2*(self.x.T).dot(self.y - Y_prediction)) / self.m 
        db = (-2*np.sum(self.y - Y_prediction)) / self.m

        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
    
    def predict(self, x):
        
        return x.dot(self.w) + self.b

