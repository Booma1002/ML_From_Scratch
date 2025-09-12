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
- [ ] K-Nearest Neighbors (KNN)
- [ ] K-Means Clustering

### Phase 2: Ensemble Methods
- [ ] Decision Tree (CART)
- [ ] Random Forest

### Phase 3: Neural Network Fundamentals
- [ ] The Neuron Layer & Activation Functions
- [ ] The Full Network & Backpropagation

---

## Completed: `Linear_Regression`

The first algorithm is complete. This is a from-scratch implementation of a Linear Regression model trained with batch gradient descent.

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

### Features Implemented:
* **Sigmoid Activation**: Converts raw linear outputs to probabilities.
* **Binary Cross-Entropy Loss (Log Loss)**: Used as the cost function.
* **Prediction**: Classifies data points with a configurable confidence threshold.
* **Evaluation**: Includes .accuracy() and .BCE() methods.
* **API**: Follows Scikit-learn style with .fit() and .predict().


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
* **The importance of Binary Cross-Entropy loss**: Unlike MSE, it punishes confident wrong predictions heavily.
---

## Next Up: K-Nearest Neighbors (KNN)

The next challenge on the gauntlet is **K-Nearest Neighbors (KNN)**.  It's a non-parametric, instance-based model. it doesn't learn a formula, instead, it memorizes the whole dataset.
