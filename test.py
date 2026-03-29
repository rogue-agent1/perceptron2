from perceptron2 import Perceptron, MulticlassPerceptron
# OR gate (linearly separable)
X = [[0,0],[0,1],[1,0],[1,1]]; y = [0,1,1,1]
p = Perceptron(lr=0.5, epochs=100).fit(X, y)
assert p.predict(X) == [0,1,1,1]
assert p.score(X, y) == 1.0
# AND gate
X2 = [[0,0],[0,1],[1,0],[1,1]]; y2 = [0,0,0,1]
p2 = Perceptron(lr=0.5, epochs=100).fit(X2, y2)
assert p2.predict(X2) == [0,0,0,1]
# Multiclass
mp = MulticlassPerceptron(lr=0.5, epochs=100)
mp.fit([[0,0],[5,5],[10,0]], [0,1,2])
assert mp.predict_one([0,0]) == 0
print("perceptron2 tests passed")
