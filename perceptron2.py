#!/usr/bin/env python3
"""Perceptron classifier. Zero dependencies."""

class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr; self.epochs = epochs; self.weights = []; self.bias = 0

    def fit(self, X, y):
        d = len(X[0])
        self.weights = [0.0]*d; self.bias = 0.0
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict_one(xi)
                if pred != yi:
                    errors += 1
                    update = self.lr * (yi - pred)
                    for j in range(d): self.weights[j] += update * xi[j]
                    self.bias += update
            if errors == 0: break
        return self

    def predict_one(self, x):
        return 1 if sum(w*xi for w, xi in zip(self.weights, x)) + self.bias >= 0 else 0

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def score(self, X, y):
        return sum(1 for p, t in zip(self.predict(X), y) if p == t) / len(y)

class MulticlassPerceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr; self.epochs = epochs; self.classifiers = {}

    def fit(self, X, y):
        classes = list(set(y))
        for c in classes:
            binary_y = [1 if yi == c else 0 for yi in y]
            self.classifiers[c] = Perceptron(self.lr, self.epochs).fit(X, binary_y)
        return self

    def predict_one(self, x):
        scores = {c: sum(w*xi for w, xi in zip(p.weights, x))+p.bias for c, p in self.classifiers.items()}
        return max(scores, key=scores.get)

    def predict(self, X): return [self.predict_one(x) for x in X]

if __name__ == "__main__":
    X = [[0,0],[0,1],[1,0],[1,1]]; y = [0,1,1,1]  # OR gate
    p = Perceptron().fit(X, y)
    print(f"OR gate: {p.predict(X)}")
