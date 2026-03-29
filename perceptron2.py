#!/usr/bin/env python3
"""perceptron2 - Perceptron classifier."""
import sys,argparse,json,random
class Perceptron:
    def __init__(self,n_features,lr=0.1):
        self.weights=[0]*n_features;self.bias=0;self.lr=lr
    def predict(self,x):return 1 if sum(w*xi for w,xi in zip(self.weights,x))+self.bias>0 else 0
    def train(self,X,y,epochs=100):
        history=[]
        for ep in range(epochs):
            errors=0
            for xi,yi in zip(X,y):
                pred=self.predict(xi)
                if pred!=yi:
                    for j in range(len(self.weights)):self.weights[j]+=self.lr*(yi-pred)*xi[j]
                    self.bias+=self.lr*(yi-pred);errors+=1
            history.append({"epoch":ep,"errors":errors})
            if errors==0:break
        return history
def main():
    p=argparse.ArgumentParser(description="Perceptron")
    p.add_argument("--task",choices=["and","or","nand"],default="and")
    p.add_argument("--epochs",type=int,default=100)
    args=p.parse_args()
    X=[[0,0],[0,1],[1,0],[1,1]]
    tasks={"and":[0,0,0,1],"or":[0,1,1,1],"nand":[1,1,1,0]}
    y=tasks[args.task]
    pc=Perceptron(2)
    history=pc.train(X,y,args.epochs)
    preds=[pc.predict(x) for x in X]
    print(json.dumps({"task":args.task,"weights":[round(w,4) for w in pc.weights],"bias":round(pc.bias,4),"predictions":dict(zip(["00","01","10","11"],preds)),"epochs_needed":len(history),"converged":history[-1]["errors"]==0},indent=2))
if __name__=="__main__":main()
