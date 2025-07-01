## Project Overview

The goal of this project is to:
- Load and preprocess a classification dataset (Iris).
- Normalize feature values.
- Train and evaluate a K-Nearest Neighbors (KNN) classifier.
- Experiment with different values of **K** (number of neighbors).
- Evaluate the model with accuracy and confusion matrix.
- Visualize the decision boundaries.

---

## Steps

### 1. Dataset Selection & Normalization

- **Dataset:** [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) (included in `sklearn.datasets`).
- **Normalization:** Use `StandardScaler` or `MinMaxScaler` from `sklearn.preprocessing` to scale the features for better KNN performance.

### 2. Model Selection: KNeighborsClassifier

- Use `KNeighborsClassifier` from `sklearn.neighbors`.
- Fit the classifier with the training data.

### 3. Experiment with Different K values

- Try different values of **K** (e.g., 1, 3, 5, 7, 9).
- Observe how the choice of K affects model performance.

### 4. Model Evaluation

- **Accuracy Score:** Use `sklearn.metrics.accuracy_score` to measure overall correctness.
- **Confusion Matrix:** Use `sklearn.metrics.confusion_matrix` to analyze class-wise performance.

### 5. Visualize Decision Boundaries

- Plot the decision boundaries of the KNN classifier for different values of K.
- Use matplotlib to visualize how the classifier separates different classes in the feature space.

---

## Requirements

- Python 3.x
- numpy
- matplotlib
- scikit-learn

Install requirements with:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Deliverables

- Well-documented code (following the above workflow).
- Plots of decision boundaries for several K values.
- Accuracy and confusion matrices for each K.

---

## References

- [scikit-learn KNeighborsClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Visualization of Decision Boundaries](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)
- [Iris Dataset Info](https://en.wikipedia.org/wiki/Iris_flower_data_set)

---

tutorial :https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbor-algorithm-in-python/
