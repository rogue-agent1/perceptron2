#!/usr/bin/env python3
"""Perceptron — single-layer and averaged perceptron."""
import random, sys

class Perceptron:
    def __init__(self, n_features):
        self.weights = [0.0]*n_features; self.bias = 0.0; self.errors = []
    def predict(self, x):
        return 1 if sum(w*xi for w,xi in zip(self.weights, x)) + self.bias > 0 else 0
    def fit(self, X, y, epochs=100, lr=1.0):
        for epoch in range(epochs):
            errs = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi); err = yi - pred
                if err != 0:
                    errs += 1
                    for j in range(len(self.weights)): self.weights[j] += lr * err * xi[j]
                    self.bias += lr * err
            self.errors.append(errs)
            if errs == 0: break
    def accuracy(self, X, y):
        return sum(self.predict(x)==yi for x,yi in zip(X,y))/len(y)

class AveragedPerceptron(Perceptron):
    def fit(self, X, y, epochs=100, lr=1.0):
        cached_w = [0.0]*len(self.weights); cached_b = 0.0; c = 1
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                if pred != yi:
                    for j in range(len(self.weights)):
                        self.weights[j] += lr * (yi - pred) * xi[j]
                        cached_w[j] += lr * c * (yi - pred) * xi[j]
                    self.bias += lr * (yi - pred)
                    cached_b += lr * c * (yi - pred)
                c += 1
        n = len(X) * epochs
        self.weights = [w - cw/n for w, cw in zip(self.weights, cached_w)]
        self.bias -= cached_b / n

if __name__ == "__main__":
    random.seed(42); X, y = [], []
    for _ in range(100):
        x = [random.uniform(-5,5), random.uniform(-5,5)]
        X.append(x); y.append(1 if x[0]*2+x[1]*3 > 0 else 0)
    p = Perceptron(2); p.fit(X[:80], y[:80])
    ap = AveragedPerceptron(2); ap.fit(X[:80], y[:80])
    print(f"Perceptron: acc={p.accuracy(X[80:],y[80:])*100:.0f}%")
    print(f"Averaged:   acc={ap.accuracy(X[80:],y[80:])*100:.0f}%")
