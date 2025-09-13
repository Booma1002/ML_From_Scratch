# The Gauntlet: A Roadmap to Machine Learning from First Principles

## Introduction

This road-map outlines a series of challenges designed to build a deep, fundamental understanding of core machine learning algorithms. The objective is to implement each model from scratch using only NumPy, thereby demystifying the "black boxes" of common libraries. Each stage includes the necessary mathematical prerequisites and a blueprint for the required class implementation.

---

## Phase 1: The Core Mechanisms

This phase focuses on building the foundational engines of machine learning, covering both regression and classification, as well as parametric and non-parametric models.

### 1. Linear Regression

**Objective:** To model the linear relationship between a set of independent features and a continuous dependent variable.

<img width="600" height="300" alt="Error_Linear_Regression" src="https://github.com/user-attachments/assets/13b0d133-390e-4b43-a367-9bd27516e326" />


**Core Logic:** The model learns a set of weights and a bias term that define a hyperplane of best fit for the data. Learning is achieved by iteratively minimizing a cost function (Mean Squared Error) using an optimization algorithm (Gradient Descent).

**Mathematical Prerequisites:**
- Linear Algebra: Vector & Matrix Operations (especially the Dot Product).
- Calculus: Partial Derivatives, The Gradient.

**Class Implementation Blueprint (`LinearRegression`):**
- `__init__(self, learning_rate, n_iterations)`: Initializes hyperparameters.
- `fit(self, X, y)`: Initializes weights and bias based on the number of features in `X`. Executes the training loop.
- `predict(self, X)`: Takes new data `X`, applies the learned weights and bias (`y_pred = X•W + b`), and returns the continuous predictions.
- `_train(self)`: (Private) A loop that repeatedly calculates predictions, computes the gradients of the cost function with respect to `W` and `b`, and updates the parameters.

### 2. Logistic Regression

**Objective:** To predict the probability that an observation belongs to one of two classes (binary classification).

<img width="600" height="300" alt="unnamed" src="https://github.com/user-attachments/assets/ea433f62-4c3f-4b42-87b8-873344c32bc9" />



**Core Logic:** Similar to Linear Regression, it computes a linear sum of the inputs. However, this output is then passed through a Sigmoid (logistic) function, which squashes the value into a probability between 0 and 1. The model learns by minimizing the Binary Cross-Entropy (Log Loss) cost function.

**Mathematical Prerequisites:**
- Functions: The Sigmoid (Logistic) Function.
- Information Theory: Basic understanding of Log Loss / Binary Cross-Entropy.

**Class Implementation Blueprint (`LogisticRegression`):**
- `__init__(self, learning_rate, n_iterations)`: Initializes hyperparameters.
- `fit(self, X, y)`: Initializes parameters and runs the training loop. The gradient update formulas are mathematically identical to Linear Regression's.
- `_sigmoid(self, z)`: (Private) A helper function implementing the sigmoid formula.
- `predict_proba(self, X)`: Takes new data, computes the linear sum, and passes it through the sigmoid to return the raw probability.
- `predict(self, X)`: Calls `predict_proba` and applies a 0.5 threshold to return the final binary class prediction (0 or 1).

### 3. K-Nearest Neighbors (KNN)

**Objective:** To classify a new data point based on the majority class of its 'k' closest neighbors in the feature space.

![0_KxkMe86skK9QRcJu](https://github.com/user-attachments/assets/7da08a82-3ead-4fb1-b343-5c2d24ae1040)


**Core Logic:** A non-parametric, instance-based algorithm. It makes no assumptions about the data distribution. The `fit` step simply involves memorizing the entire training dataset. Prediction involves calculating distances, finding the nearest neighbors, and taking a majority vote.

**Mathematical Prerequisites:**
- Linear Algebra: Vectors, Norms.
- Geometry: Euclidean Distance.

**Class Implementation Blueprint (`KNN_Classifier`):**
- `__init__(self, k)`: Initializes the number of neighbors, `k`.
- `fit(self, X, y)`: Stores the training data `X_train` and `y_train`.
- `predict(self, X_test)`: Iterates through each point in `X_test`. For each point, it calculates the distance to all points in `X_train`, finds the `k` nearest neighbors, and predicts the class based on a majority vote of their labels.
- `_euclidean_distance(self, p1, p2)`: (Private) A helper function to compute the distance between two vectors.

### 4. K-Means Clustering

**Objective:** To partition a dataset into 'k' distinct, non-overlapping clusters, where each data point belongs to the cluster with the nearest mean (centroid).

<img width="600" height="300" alt="llustration-of-K-means-Clustering-Note-The-estimation-routine-of-K-means-involves-i" src="https://github.com/user-attachments/assets/f1d54d57-7bb7-4e73-a9f8-9f9aba1f94da" />


**Core Logic:** An unsupervised learning algorithm that iteratively refines cluster assignments. It begins by randomly placing 'k' centroids. Then, it alternates between two steps: (1) assigning each data point to its closest centroid, and (2) updating each centroid to be the mean of its newly assigned points. This process repeats until the cluster assignments no longer change.

**Mathematical Prerequisites:**
- Statistics: Mean (Average).
- Geometry: Euclidean Distance.

**Class Implementation Blueprint (`KMeans`):**
- `__init__(self, k, max_iters)`: Initializes the number of clusters `k` and a maximum number of iterations.
- `fit(self, X)`: Initializes `k` random centroids. Executes the iterative process of assigning points and updating centroids until convergence.
- `predict(self, X)`: Takes new data and assigns each point to the closest of the final, trained centroids.

---

## Phase 2: Ensemble Methods

This phase focuses on combining multiple simple models to create a single, more powerful, and robust model.

### 5. Decision Tree (CART)

**Objective:** To create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

<img width="600" height="300" alt="a-Decision-tree-graph-form-node-definitions-and-space-divisions-b-Impurity-function" src="https://github.com/user-attachments/assets/a81aedcf-6cf0-4d0b-b2f7-7861704c6eae" />


**Core Logic:** The algorithm recursively splits the dataset into smaller and smaller subsets based on the feature that provides the "best split." The best split is determined by the one that maximizes information gain (or minimizes impurity). The recursion stops when a node is pure (all samples belong to one class) or another stopping criterion is met.

**Mathematical Prerequisites:**
- Set Theory: Subsets.
- Information Theory: Gini Impurity or Entropy.

**Class Implementation Blueprint (`DecisionTree`):**
- `__init__(self, max_depth, min_samples_split)`: Initializes stopping criteria.
- `fit(self, X, y)`: Initiates the recursive tree-building process.
- `predict(self, X)`: Traverses the trained decision tree for each sample in `X` to arrive at a leaf node prediction.
- `_grow_tree(self, X, y, depth)`: (Private) The core recursive function. It finds the best split, splits the data, and calls itself on the resulting child nodes.
- `_best_split(self, X, y)`: (Private) A function that iterates through all features and thresholds to find the split that results in the lowest impurity.
- `_impurity(self, y)`: (Private) A function to calculate Gini impurity for a set of labels.

### 6. Random Forest

**Objective:** To improve the predictive accuracy and control over-fitting by creating an ensemble of Decision Trees and outputting the mode of the classes (classification) or mean prediction (regression) of the individual trees.

<img width="600" height="300" alt="Random-Forest-1500x800-v3" src="https://github.com/user-attachments/assets/b4465cd7-f731-4bd6-be70-1ca8e52853c7" />


**Core Logic:** It builds multiple Decision Trees on different bootstrapped samples of the training data. Additionally, when splitting a node, it only considers a random subset of features. This "double randomness" creates a diverse set of trees whose combined prediction is more accurate and robust.

**Mathematical Prerequisites:**
- Statistics: Bootstrapping (Sampling with Replacement), Majority Vote (Mode).

**Class Implementation Blueprint (`RandomForest`):**
- `__init__(self, n_trees, max_depth, ...)`: Initializes the number of trees and hyperparameters for the individual trees.
- `fit(self, X, y)`: Creates a loop that, for `n_trees`, generates a bootstrapped sample of the data and trains a `DecisionTree` model on it. Stores all trained trees.
- `predict(self, X)`: Passes each sample in `X` through every tree in the forest, collects all the predictions, and returns the majority vote for each sample.

---

## Phase 3: Neural Network Fundamentals

This phase deconstructs the core components of neural networks to understand how they learn complex patterns.

### 7. Neural Network

**Objective:** To build a simple, fully-connected neural network for classification that learns through the process of backpropagation.

<img width="600" height="300" alt="1_OGFvJgMe21_5fCzUUyLwrw" src="https://github.com/user-attachments/assets/e84706eb-3d2d-4457-8372-23795b66411a" />


**Core Logic:** A network is composed of layers of interconnected "neurons." Data is passed forward through the network (the forward pass), with each layer performing a linear transformation followed by a non-linear activation function. The final output is compared to the true label via a loss function. The error is then propagated backward through the network (backpropagation), calculating the gradient of the loss with respect to every weight. These gradients are then used to update the weights via gradient descent.

**Mathematical Prerequisites:**
- Linear Algebra: All matrix operations.
- Calculus: The Chain Rule (critically important), Derivatives of common functions (Sigmoid, ReLU).
- Information Theory: Cross-Entropy Loss.

**Class Implementation Blueprint (`NeuralNetwork`):**
- `__init__(self, layer_sizes, learning_rate, ...)`: Initializes the network architecture (number of layers and neurons per layer), hyperparameters, and weights.
- `fit(self, X, y)`: The main training loop. For each epoch, it performs a forward pass, computes the loss, performs a backward pass, and updates the weights.
- `predict(self, X)`: Performs a single forward pass on new data and returns the final class predictions.
- `_forward_pass(self, X)`: (Private) Pushes an input `X` through all layers, applying linear transformations and activation functions, storing the intermediate outputs for use in backpropagation.
- `_backward_pass(self, y_true, forward_cache)`: (Private) The core backpropagation logic. It starts from the final layer's error and uses the chain rule to calculate the gradients for all weights and biases in the network.
- `_update_params(self, grads)`: (Private) Applies the calculated gradients to the network's weights and biases.
