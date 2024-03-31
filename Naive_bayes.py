import numpy as np


class NaiveBayes():
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.__classes = np.unique(y)
        n_classes = len(self.__classes)
        
        self.__mean = np.zeros((n_classes,n_features),dtype=np.float64)
        self.__var = np.zeros((n_classes,n_features),dtype=np.float64)
        self.__priors = np.zeros(n_classes,dtype=np.float64)

        for idx,c in enumerate(self.__classes):
            X_c = X[y == c]
            self.__mean[idx,:] = X_c.mean(axis = 0) 
            self.__var[idx,:] = X_c.var(axis=0)
            self.__priors[idx] = X_c.shape[0]/float(n_samples)
            
               
        
    def predict(self,X):
        y_pred = [self.__predict(x) for x in X]
        return np.array(y_pred)
    
    def __predict(self,x):
        posteriors = []
        for idx,c in enumerate(self.__classes):
            prior = np.log(self.__priors[idx])
            posterior = np.sum(np.log(self.__pdf(idx,x)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
            
        return self.__classes[np.argmax(posteriors)] 
        
    def __pdf(self,class_idx,x):
        mean = self.__mean[class_idx]
        var = self.__var[class_idx]
        numerator = np.exp(-(x-mean)**2)/(2*var)
        denominator = np.sqrt(2 * np.pi *var)
        return numerator / denominator
    
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
    model = NaiveBayes()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    accurancy = acc(y_test,predictions)
    print(accurancy)
    # model = GaussianNB()
    # model.fit(X_train,y_train)
    # predictions = model.predict(X_test)
    # accs = acc(y_test,predictions)
    # print(accs)