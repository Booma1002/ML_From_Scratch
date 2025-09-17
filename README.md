# ML Algorithms From Scratch
I am learning Machine Learning by implementing the core algorithms from scratch using only NumPy.

## Project Overview

This repository is my personal journey into the heart of machine learning. The goal is to build the fundamental ML algorithms from the ground up using only **NumPy**. This is not about creating a production-ready library, but about dismantling the "black boxes" of popular frameworks to gain a deep, first-principles understanding of how they work.

This is my gauntlet.

---

## The Gauntlet: Implemented Algorithms

Here is the checklist of algorithms I am building.

### Phase 1: The Core Mechanisms
- [x] **Linear Regression**
- [x] Logistic Regression
- [x] K-Nearest Neighbors (KNN)
- [x] K-Means Clustering

### Phase 2: Ensemble Methods
- [ ] Decision Tree (CART)
- [ ] Random Forest

### Phase 3: Neural Network Fundamentals
- [ ] The Neuron Layer & Activation Functions
- [ ] The Full Network & Backpropagation

---

## Completed: `Linear_Regression`

The first algorithm is complete. This is a from-scratch implementation of a Linear Regression model trained with batch gradient descent.
[See the class source code here](https://github.com/Booma1002/ML_From_Scratch/blob/main/Hazem_Gamal.py#L2)

### Features Implemented:
* **Training:** Uses Gradient Descent to optimize weights.
* **Evaluation:** Includes an `score` method that calculates the R2  (Coefficient of Determination) of **Ordinary Least Squares Regression**.
* **API:** Follows a Scikit-learn style API with `.fit()` and `.predict()` methods.
* **Convergence:** Stops training early if the R2 score stops improving significantly.

### Usage Example

Here's how to use the `Linear_Regression` class on a 2D dataset:

```python
# import the class from the library
from Hazem_Gamal import Linear_Regression
import numpy as np

# 1. Create some sample data
# X should have a shape of [n_samples, n_features]
# Reshape X if necessary
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 5])

# 2. initialize and train the model
model = Linear_Regression(learning_rate=0.01, n_iterations=100)
model.fit(X_train, y_train)

# 3. make predictions on new data
X_new = np.array([[6], [7]])
y_new = np.array([[6], [7]]) # suppose this is the label
predictions = model.predict(X_new)
print(f"predictions of the data: {predictions}")

# 4. Score the model
score = model.score(y_new, predictions)
print(f"The R2 score is {score:0.4f}")

# Plot to visualize:
plt.scatter(X_new, y_new)
plt.title(f'R2 = {score:0.2f}')
sns.lineplot(x=X_new.reshape(-1), y=predictions,color='red', linewidth=2)
plt.grid(True)
plt.show()
```
```
Output:
Iteration 100: Final R2_score = 0.3327
predictions of the data: [6.67565479 7.6420862 ]
The R2 score is -6.3409
```
<img width="544" height="433" alt="image" src="https://github.com/user-attachments/assets/98e21d59-3cfb-43c4-87dd-3b9201dee33e" />


### What I Learned

* **The critical importance of matrix dimensionality**: Every bug I faced came from getting the shapes wrong, which forced me to understand the linear algebra behind the dot product.
* **Why write a well-formatted code**: I happened to use untidy structure at the beginning, it made resolving errors much harder. When I refined the code i found that about 40% of it was overwritten and leads to overwhelming time waste. Next time I'll keep this in mind.
* **How Gradient Descent Actually Works**: The main idea is to minimize a cost function by iteratively updating parameters, and stopping when it converges.

---
## Completed: `Logistic_Regression`
The second algorithm is a from-scratch implementation of Logistic Regression, trained with gradient descent and optimized using Binary Cross-Entropy Loss.
[See the class source code here](https://github.com/Booma1002/ML_From_Scratch/blob/main/Hazem_Gamal.py#L72)
### Features Implemented:
* **Sigmoid Activation**: Converts raw linear outputs to probabilities.
* **Binary Cross-Entropy Loss (Log Loss)**: Used as the cost function.
* **Prediction**: Classifies data points with a configurable confidence threshold.
* **Evaluation**: Includes `.accuracy()` and `.BCE()` methods.
* **API**: Follows Scikit-learn style with `.fit()` and `.predict()`.


### Usage Example

Here's how to use the `Logistic_Regression` class on a classification dataset:

```python
from Hazem_Gamal import Logistic_Regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_classification(n_samples=10000, n_features=4, n_classes=2, random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

# Train
model = Logistic_Regression(n_iterations=10000, learning_rate=0.001, convergence_threshold=0.0,history_step=250)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc =model.accuracy(y_test, y_pred)
loss =model.BCE(y_test, y_pred)

# Evaluate
print("Accuracy:", acc)
print("Loss:", loss)


# Visualize loss curve:
sns.set_style("darkgrid")
sns.lineplot(model.history_acc, marker='o',label=f"Training Accuracy: {model.history_acc[-1]/100:.2f}")
sns.lineplot(model.history_loss, marker='s',color='r',label=f'Loss: {model.history_loss[-1]/100:.2f}')
plt.title("Training Progress")
plt.axhline(y=100, color='black',label='1.00')
plt.xlabel(f"Iteration*({model.history_step})")
plt.ylabel("Metric")
plt.legend()
plt.show()
```
```
Output:
Iteration 10000: Final Log_Loss_score = 0.5383
Accuracy: 0.8686666666666667
Loss: 2.721655582446835
```
<img width="568" height="453" alt="image" src="https://github.com/user-attachments/assets/8d87bfda-7d69-48f5-b80d-473709bbc8d9" />

### What I Learned


* **Gradient Descent in practice**: Iterative updates converge to the optimal solution if learning rate and stopping conditions are chosen well.
* **Why Logistic Regression is a linear classifier**: Even though it outputs probabilities, the decision boundary is linear.
* **The importance of Binary Cross-Entropy loss**: Unlike `MSE`, it punishes confident wrong predictions heavily.
---

## Completed: `K_Nearest_Neighbors`
The third algorithm is a from-scratch implementation of K-Nearest Neighbors (KNN). This is a non-parametric, instance-based model that classifies new data points based on the majority class of its 'k' nearest neighbors.
[See the class source code here](https://github.com/Booma1002/ML_From_Scratch/blob/main/Hazem_Gamal.py#L166)

### Features Implemented:
* **Instance-Based Learning:** The model memorizes the entire training dataset instead of learning weights.

* **Euclidean Distance:** Uses the `L2` norm to calculate the distance between data points in a multi-dimensional space.

* **Efficient Majority Vote:** Implemented a fast majority class function using NumPy's `np.unique(return_counts =True)` and `argmax` for performance, avoiding slower external libraries for the core logic.

* **Modular Prediction:** Split the prediction logic into `_predict_single` for one sample and a public `predict` method that iterates over it, improving code clarity and maintainability.

* **API:** Follows Scikit-learn style with `.fit()`, `.predict()`, and `.accuracy()` methods.
  

### Usage Example

Here's how to use the `K_Nearest_Neighbors` class on a multi-class classification problem:

```python
from Hazem_Gamal import K_Nearest_Neighbors as KNN
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


X, y = make_blobs(n_samples=1000, centers=4, n_features=2, cluster_std=1.3, random_state=37)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=37)

k=51
model = KNN(k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.accuracy(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(6, 4))
plt.style.use('ggplot')
sns.scatterplot(
    x=X_test[:, 0],
    y=X_test[:, 1],
    hue=y_test,
    palette='plasma',
    s=80,
    edgecolor='k',
    alpha=0.7,
)

plt.title(f'KNN decision boundary (k={k})\nAccuracy = {accuracy*100:.2f}%', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend(title='Actual Class')
plt.grid(False)
plt.show()
```
```
Output:
Model Accuracy: 97.00%
```
<img width="544" height="433" alt="image" src="https://github.com/user-attachments/assets/4cb9138e-f47e-4789-bf81-fdaa26b1e403" />

### What I Learned

* **The Geometry of Distance Metrics:** Understanding that the `L2` (Euclidean) norm effectively finds neighbors within a circle was key. It sparked curiosity about how other norms (like L-infinity) would create different decision boundaries (like a square).

* **NumPy-First Optimization:** My initial thought was to use scipy.stats.mode, but sticking to the "NumPy only" rule forced me to find a more fundamental and faster solution using `np.unique(return_counts =True)` and `np.argmax`. This was a great lesson in vectorized computation.

* **The Power of Code Modularity:** Trying to write a single predict function was getting complicated. Splitting the logic into a helper function (`_predict_single`) that handles one point and a main function that loops over it made the code much cleaner and easier to debug.

* **Lazy Learning Trade-offs:** KNN is simple to understand and implement because it doesn't have a "training" phase. However, this means prediction is computationally expensive, as it has to compare a new point to every single point in the training data.
---



## Completed: `K_Means`
The fourth algorithm is a from-scratch implementation of K-Means clustering. This is an unsupervised learning model that partitions a dataset into 'k' distinct, non-overlapping clusters. The task was to build a model that can discover hidden groups in unlabeled data by iteratively assigning each point to its nearest centroid and then updating the centroid's value to the mean of its newly assigned points.
[See the class source code here](https://github.com/Booma1002/ML_From_Scratch/blob/main/Hazem_Gamal.py#L198)

### Features Implemented:
* **Unsupervised Learning:** The model learns clusters from features (`X`) alone, without needing any labels (`y`).

* **Iterative Optimization:** Implements the two-step Expectation-Maximization process: the assignment step (assigning points to the nearest centroid) and the update step (re-calculating centroids from the mean of assigned points).

* **Vectorized Centroid Updates:** Uses `np.mean(axis=0)` for efficient and generalizable centroid calculations, allowing the model to work on data with any number of features.

* **Evaluation:** Includes a `Score` method that correctly implements the Silhouette Score from scratch, a common metric for evaluating the quality of clusters.
<img width="538" height="395" alt="image" src="https://github.com/user-attachments/assets/487bfad5-2266-47a6-b54b-e703228b9e95" />


* **API:** Follows Scikit-learn style with `.fit()` and `.predict()` methods.
  

### Usage Example

Here's how to use the `K_Means` class on a clustering problem:

```python
from Hazem_Gamal import K_Means
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

X,y = make_blobs(n_samples=5000,cluster_std=1.5, centers=5,n_features=2, random_state=38)
X_train, X_test, _,_ = train_test_split(X,y,test_size=0.1, random_state=38)

model = K_Means(k=5,max_iter=20,threshold=0.00000001)
model.fit(X_train)
y_pred = model.predict(X_test)


plt.figure(figsize=(6, 4))
plt.style.use('ggplot')
sns.scatterplot(
    x=X_test[:, 0],
    y=X_test[:, 1],
    hue=y_pred,
    palette='Set1',
    s=80,
    edgecolor='k',
    alpha=0.7,
)
plt.title(f'K-Means Silhouette Score: {model.Score(X_test):.2f}')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend(title='Clusters',)

plt.grid(axis='x',linestyle='--',linewidth=0.7,color='blue')
plt.grid(axis='y',linestyle='--',linewidth=0.7,color='grey')
plt.show()
```
Output:
```
Converged at iteration 10
```
<img width="538" height="395" alt="image" src="https://github.com/user-attachments/assets/9ff7adbc-fbaf-415a-bca2-09ba5b743a95" />


### What I Learned
* **The Power of Vectorized Operations:** The importance of using `np.mean(axis=0)` was a critical lesson. It replaced a complex, hardcoded loop, making the code cleaner, faster, and versatile enough to handle any number of dimensions. This was a key insight into writing algorithms, not just scripts.

* **The Importance of Research for Evaluation:** There is no "accuracy" in unsupervised learning. This forced me to research evaluation methods like the `Silhouette Score`. Implementing it from scratch cemented my understanding of how to measure a model's performance based on its own output (cluster cohesion and separation).

* **The Problem of Local Minima:** During debugging, I observed that the algorithm's final centroids could vary significantly based on their random starting positions. Research confirmed this is a known issue with K-Means (sticking in local minima). This taught me that even a converged model is not guaranteed to be the optimal one, and techniques like `K-Means++` exist to mitigate this.

* **Reusable Design Patterns:** The value of helper functions like `__euclidean_distance` became very clear. Building it for KNN and reusing it for K-Means showed how multipurpose, modular functions are a core part of an efficient workflow.

---



## Next Up: Decision Tree (CART)
The next challenge on the gauntlet is the Decision Tree (CART). This marks the beginning of Phase 2 and a fundamental shift from mathematical optimization (like gradient descent) to a new paradigm: recursive logic and information theory. The goal is to build a model that learns by discovering the most informative if/else questions to ask of the data. This is the foundational building block for powerful ensemble methods like Random Forest.

