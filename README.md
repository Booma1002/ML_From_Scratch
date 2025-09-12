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
- [ ] Logistic Regression
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
<img width="544" height="433" alt="image" src="https://github.com/user-attachments/assets/98e21d59-3cfb-43c4-87dd-3b9201dee33e" />


### What I Learned

* **The critical importance of matrix dimensionality**: Every bug I faced came from getting the shapes wrong, which forced me to understand the linear algebra behind the dot product.
* **Why write a well-formatted code**: I happened to use untidy structure at the beginning, it made resolving errors much harder. When I refined the code i found that about 40% of it was overwritten and leads to overwhelming time waste. Next time I'll keep this in mind.
* **How Gradient Descent Actually Works**: The main idea is to minimize a cost function by iteratively updating parameters, and stopping when it converges.

---

## Next Up: Logistic Regression

The next challenge on the gauntlet is **Logistic Regression**. This will be my first from-scratch classifier and will require implementing the Sigmoid function and the Binary Cross-Entropy loss function.
