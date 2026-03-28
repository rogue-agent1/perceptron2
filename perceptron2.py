#!/usr/bin/env python3
"""perceptron2 - Multi-layer perceptron from scratch."""
import sys,math,random
def sigmoid(x):return 1/(1+math.exp(-max(-500,min(500,x))))
def sigmoid_d(x):s=sigmoid(x);return s*(1-s)
class MLP:
    def __init__(s,layers):
        s.W=[];s.b=[]
        for i in range(len(layers)-1):
            s.W.append([[random.gauss(0,0.5) for _ in range(layers[i])] for _ in range(layers[i+1])])
            s.b.append([0]*layers[i+1])
    def forward(s,x):
        s.activations=[x];s.zs=[]
        for W,b in zip(s.W,s.b):
            z=[sum(W[j][k]*x[k] for k in range(len(x)))+b[j] for j in range(len(b))]
            x=[sigmoid(zi) for zi in z];s.zs.append(z);s.activations.append(x)
        return x
    def backward(s,target,lr=0.1):
        deltas=[None]*len(s.W)
        output=s.activations[-1]
        deltas[-1]=[(output[j]-target[j])*sigmoid_d(s.zs[-1][j]) for j in range(len(output))]
        for l in range(len(s.W)-2,-1,-1):
            deltas[l]=[sum(deltas[l+1][j]*s.W[l+1][j][i] for j in range(len(deltas[l+1])))*sigmoid_d(s.zs[l][i]) for i in range(len(s.zs[l]))]
        for l in range(len(s.W)):
            for j in range(len(s.W[l])):
                for k in range(len(s.W[l][j])):s.W[l][j][k]-=lr*deltas[l][j]*s.activations[l][k]
                s.b[l][j]-=lr*deltas[l][j]
    def train(s,X,y,epochs=1000,lr=0.1):
        for _ in range(epochs):
            for xi,yi in zip(X,y):s.forward(xi);s.backward(yi,lr)
if __name__=="__main__":
    mlp=MLP([2,4,1]);X=[[0,0],[0,1],[1,0],[1,1]];y=[[0],[1],[1],[0]]
    mlp.train(X,y,epochs=5000,lr=0.5)
    print("XOR problem:");
    for xi,yi in zip(X,y):pred=mlp.forward(xi);print(f"  {xi} → {pred[0]:.3f} (expected {yi[0]})")
