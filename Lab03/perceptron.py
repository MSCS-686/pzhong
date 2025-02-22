from classifier import classifier
import numpy as np

class perceptron(classifier):

    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, Y):
        self.weights = np.zeros(X.shape[1]+1)
        for _ in range(self.epochs):
            for x, y in zip(X, Y):
                s = np.dot(x, self.weights[1:]) + self.weights[0]
                hyp = self.activation(s)
                self.weights[1:] += self.learning_rate * (y - hyp) * x
                self.weights[0] += self.learning_rate * (y - hyp)

    def predict(self, X):
        hyps = []
        for x in X:
            hyps.append(self.predict_one(x))
        return hyps

    def predict_one(self, x):
        s = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(s)
        
    def activation(self, x):
        result = self.sigmoid(x)
        if result > 0.5:
            return 1
        else:
            return 0

    def get_weights(self):
        return self.weights[1:]

    def get_bias(self):
        return self.weights[0]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))