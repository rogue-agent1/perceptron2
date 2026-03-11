#!/usr/bin/env python3
"""Perceptron — single-layer binary classifier."""
import sys, random

class Perceptron:
    def __init__(self, n_features, lr=0.1):
        self.w = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.b = 0.0; self.lr = lr
    def predict(self, x):
        return 1 if sum(wi*xi for wi, xi in zip(self.w, x)) + self.b > 0 else 0
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                if pred != yi:
                    errors += 1
                    for j in range(len(self.w)):
                        self.w[j] += self.lr * (yi - pred) * xi[j]
                    self.b += self.lr * (yi - pred)
            if errors == 0: return epoch + 1
        return epochs
    def score(self, X, y):
        return sum(self.predict(xi) == yi for xi, yi in zip(X, y)) / len(y)

if __name__ == "__main__":
    random.seed(42)
    # AND gate
    X = [[0,0],[0,1],[1,0],[1,1]]; y = [0,0,0,1]
    p = Perceptron(2); epochs = p.fit(X, y)
    print(f"AND gate: converged in {epochs} epochs, acc={p.score(X,y):.0%}")
    # OR gate
    X = [[0,0],[0,1],[1,0],[1,1]]; y = [0,1,1,1]
    p = Perceptron(2); epochs = p.fit(X, y)
    print(f"OR gate:  converged in {epochs} epochs, acc={p.score(X,y):.0%}")
    # Linear classification
    X = [[random.gauss(0,1), random.gauss(0,1)] for _ in range(50)] + \
        [[random.gauss(3,1), random.gauss(3,1)] for _ in range(50)]
    y = [0]*50 + [1]*50
    p = Perceptron(2); p.fit(X, y, 200)
    print(f"Linear:   acc={p.score(X,y):.1%}")
