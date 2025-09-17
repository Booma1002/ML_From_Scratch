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








class K_Nearest_Neighbors:
    def __init__(self,k=3):
        self.k = k
        self.X_train =None
        self.y_train =None
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = [self._pred_single(x) for x in X_test]
        return np.array(y_pred)

    def _pred_single(self, x_test):
        distances = [self.__euclidean_distance(x_test, x_train) for x_train in self.X_train]
        k_i = np.argsort(distances)[:self.k]
        voting = [self.y_train[i] for i in k_i]
        vote = self.__majority(voting)
        return vote

    def __majority(self,votes):
        vote, cnt = np.unique(votes, return_counts=True)
        return vote[np.argmax(cnt)]

    def __euclidean_distance(self,p,q):
        return np.sqrt(np.sum((p-q)**2))

    def accuracy(self, y_test, y_pred):
        return np.mean(y_test == y_pred)


class K_Means():
    def __init__(self,k=4,max_iter=20,threshold=0.001):
        self.k = k
        self.max_iter = max_iter
        self.centroids_ = None
        self.idx = np.arange(k)
        self.points={}
        self.threshold=threshold
        self.improve=1000
        self.improvement =[]
        self.prev=1e7

    def fit(self,X):
        self.X = np.array(X)
        self.centroids_ = np.random.rand(self.k,X.shape[1])
        i=0
        while(i<self.max_iter):
            self.__assign_points()
            self.__update_centrs()
            if((self.__converged())&(i>3)):
                print(f"Converged at iteration {i}")
                break
            i+=1

    def __assign_points(self):
        points ={i:[] for i in self.idx}
        cur=1
        for p in self.X:
            dist = np.array([self.__euclidean_distance(p,c) for c in self.centroids_])
            idx = np.argmin(dist)
            points[idx].append(p)
            cur+=np.min(dist)
        self.points = points
        if(cur==0):
            cur+=.00000001
        self.improve = abs(self.prev - cur) / cur
        self.improvement.append(self.improve)
        self.prev = cur

    def __update_centrs(self):
        for i in self.points:
            if len(self.points[i]) > 0:
                self.centroids_[i] = np.mean(self.points[i], axis=0)

    def predict(self,X_test):
        X_test = np.array(X_test)
        y_pred =[np.argmin([self.__euclidean_distance(p,c) for c in self.centroids_])for p in X_test]
        return y_pred

    def __euclidean_distance(self, p, q):
        return np.sqrt(np.sum((p - q) ** 2))

    def __converged(self):
        return self.improve < self.threshold

    def __nearest_cluster(self,cluster_idx):
        dist = np.array([self.__euclidean_distance(p,self.centroids_[cluster_idx]) for p in self.centroids_])
        return np.argsort(dist)[1]

    def Score(self, X):
        scores = []
        pred = self.predict(X)
        i=0
        for point in X:
            this = pred[i]
            neighbor = self.__nearest_cluster(this)
            a = np.mean([self.__euclidean_distance(point,pi) for pi in self.points[this]])
            b=  np.mean([self.__euclidean_distance(point,ci)for ci in self.points[neighbor]])
            scores.append((b-a)/(max(b,a)+.000000001))
            i+=1
        return np.mean(scores)








