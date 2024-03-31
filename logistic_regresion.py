import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegresion():
    def __init__(self,lr =0.001,n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples , n_fiatures = X.shape
        self.weight = np.zeros(n_fiatures)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_pred = np.dot(X,self.weight) + self.bias
            predictions = sigmoid(linear_pred)
            dw = (1/n_samples) * np.dot(X.T,(predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr*db
            
            
    def predict(self,X):
        linear_pred = np.dot(X,self.weight) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    bc = datasets.load_breast_cancer()
    X,y = bc.data,bc.target
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)


    def accurancy(y_pred,predict):
        k = np.sum(y_pred == predict)/len(y_pred)
        return k

    model = LogisticRegresion()
    model.fit(X_train,y_train)
    predictios = model.predict(X_test)
    print(accurancy(y_test,predictios))

