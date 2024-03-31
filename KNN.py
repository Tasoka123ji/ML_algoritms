import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist



class KNN:
    def __init__(self,k = 3) -> None:
        self.k = k
    
    def fit(self,X_train,y_train):
        self.X = X_train
        self.y = y_train
        
    def predict(self,X):
        predictions = [self.__predict(x) for x in X]
        return predictions
    
    def __predict(self,x):
        distensis = [euclidean_distance(x,x1) for x1 in self.X]
        most_distensis = np.argsort(distensis)[:self.k]
        y_pred = [self.y[i] for i in most_distensis]
        most_comon = Counter(y_pred).most_common()
        return most_comon[0][0]
    
    

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    
    def acc(y_test,y_pred):
        return np.sum(y_test == y_pred)/len(y_test)
    
    X,y = datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=123)
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size= 0.2,
                                                        random_state=123)
    model = KNN()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    accurancy = acc(y_test,predictions)
    print(accurancy)
