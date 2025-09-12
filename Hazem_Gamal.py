import numpy as np

class Linear_Regression:
    def __init__(self,n_iterations=100, convergence_threshold =0.0010,learning_rate=0.1, loss_function='mse'):
        self.X = None
        self.y = None
        self.W = None
        self.b = None
        self.q = []
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.learning_rate = learning_rate

    def init(self,X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        n_features = self.X.shape[1]
        self.W = np.random.randn(n_features)
        self.b = 0.0



    def fit(self,X,y):
        self.init(X,y)
        self.__train()

    def __train(self):
        N = self.X.shape[0]  # number of samples
        cnt = 0
        for i in range(self.n_iterations):
            cnt += 1
            y_pred = self.predict(self.X)

            # gradients
            dW = (2 / N) * self.X.T.dot(y_pred - self.y)  # shape (n_features,)
            db = (2 / N) * np.sum(y_pred - self.y)  # scalar

            # update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # track convergence
            r2 = self.score(self.y, y_pred)
            self.q.append(r2)
            if len(self.q) > 2:
                ratio = abs(self.q[-1] - self.q[-2]) / (abs(self.q[-1]) + 1e-8)
                if ratio < self.convergence_threshold:
                    print(f"Convergence after {cnt} iterations, ratio={ratio:.6f}")
                    break
                self.q.pop(0)

        print(f"Iteration {cnt}: Final R2_score = {r2:.4f}")

    def predict(self, X):
        y_pred = X.dot(self.W) + self.b   # shape (n_samples,)
        return y_pred

    ########################
    ## Evaluation ##
    ########################
    def __rss(self, y, y_pred):
        return np.sum((y - y_pred)**2)

    def __tss(self, y, y_pred):
        y_mean = np.mean(y)
        return np.sum((y - y_mean)**2)

    def score(self, y, y_pred):
        return 1 - (self.__rss(y, y_pred) / (self.__tss(y, y_pred) + 1e-7))

    def __mse(self, y, y_pred):
        return np.mean((y - y_pred)**2)







