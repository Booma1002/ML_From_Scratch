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

    def __init(self,X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        n_features = self.X.shape[1]
        self.W = np.random.randn(n_features)
        self.b = 0.0



    def fit(self,X,y):
        self.__init(X,y)
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


class Logistic_Regression:

    def __init__(self, n_iterations=100, convergence_threshold=0.0,confidence_threshold=0.5, learning_rate=0.1,history_step =100):
        self.X = None
        self.y = None
        self.W = None
        self.b = None
        self.q = []
        self.history_loss = []
        self.history_acc = []
        self.history_step = history_step
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate

    def __init(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        n_features = self.X.shape[1]
        self.W = np.random.randn(n_features)
        self.b = 0.0

    def fit(self, X, y):
        self.__init(X, y)
        self.__train()


    def __train(self):
        N = self.X.shape[0]  # number of samples
        cnt = 0
        for i in range(self.n_iterations):
            cnt += 1
            y_pred = self.predict(self.X)
            # gradients
            dW = (1/ N) * self.X.T.dot(y_pred - self.y)  # shape (n_features,)
            db = (1 / N) * np.sum(y_pred - self.y)  # scalar

            # update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # track convergence
            bce = self.BCE(self.y, self.pred_proba(self.X))
            acc = self.accuracy(self.y, self.predict(self.X))
            self.q.append(bce)
            if(cnt % self.history_step == 0):
                self.history_loss.append(bce*100)
                self.history_acc.append(acc*100)
            if len(self.q) > 2:
                ratio = abs(self.q[-1] - self.q[-2]) / (0.5 * (abs(self.q[-1]) + abs(self.q[-2])) + 1e-8)
                if (ratio < self.convergence_threshold)&(cnt>10):
                    print(f"Convergence after {cnt} iterations, ratio={ratio:.12f}")
                    break
                self.q.pop(0)
        print(f"Iteration {cnt}: Final Log_Loss_score = {bce:.4f}")


    def predict(self, X):  # shape (n_samples,)
        probs = self.pred_proba(X)
        return (probs >= self.confidence_threshold).astype(int)

    def pred_proba(self, X):
        X = np.array(X) # shape (n_samples,)
        linear = X.dot(self.W) + self.b
        return self.__sigmoid(linear)

    def __sigmoid(self,z): # confidence
        return 1 / (1 + np.exp(-z))

    ###############
    #  Evaluation #
    ###############

    def BCE(self, y, y_pred):
        # log loss, we punish wrong predictions by big
        # sum of element-wise punishments
        # 2 possible outcomes are: y and 1-y
        # To prevent log(0), i clipped values to a range 0<pred<1
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        log_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return log_loss


    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

