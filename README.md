# Principal Component Analysis From Scratch

A NumPy-based implementation of **Principal Component Analysis (PCA)** from scratch using the covariance matrix and eigen-decomposition.

## Overview

Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique used to reduce the number of features in a dataset while preserving as much of the data's variance as possible.

This project implements the main steps of PCA without using a dedicated PCA library such as Scikit-learn.

The implementation performs:

1. Feature selection
2. Feature standardization
3. Covariance matrix calculation
4. Eigen-decomposition
5. Sorting eigenvalues and eigenvectors
6. Explained variance calculation
7. Selection of the number of principal components
8. Projection of the data into the reduced-dimensional space

## Dataset

This project uses the **Student Performance dataset**.

The dataset contains demographic, social, and academic information about students.

Only numerical features are used for PCA. The `G3` column is removed because it represents the final grade and is treated as the target variable rather than an input feature.

## Implementation

### 1. Feature Selection

The numerical columns are selected from the dataset and `G3` is removed.

```python
numeric_df = df.select_dtypes(include=["int64", "float64"])
X_df = numeric_df.drop(columns=["G3"])
X = X_df.to_numpy(dtype=float)
```

### 2. Standardization

Each feature is standardized using its mean and standard deviation:

\[
X_{std} = \frac{X-\mu}{\sigma}
\]

where:

- `μ` is the mean of the feature
- `σ` is the standard deviation of the feature

Standardization ensures that features with larger numerical scales do not dominate the PCA calculation.

### 3. Covariance Matrix

The covariance matrix is calculated from the standardized data:

```python
cov_matrix = np.cov(X_std, rowvar=False)
```

The covariance matrix describes how the different features vary with respect to one another.

### 4. Eigen-decomposition

The covariance matrix is decomposed into eigenvalues and eigenvectors:

```python
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
```

The **eigenvectors represent the principal directions**, while their corresponding **eigenvalues represent the variance along those directions**.

### 5. Sorting Eigenvalues and Eigenvectors

The eigenvalues are sorted from largest to smallest. The eigenvectors are reordered using the same indices.

The eigenvector corresponding to the largest eigenvalue becomes the first principal component.

### 6. Explained Variance

The explained variance ratio of each principal component is calculated as:

\[
\text{Explained Variance Ratio}_i =
\frac{\lambda_i}{\sum_j \lambda_j}
\]

where `λᵢ` is the eigenvalue corresponding to principal component `i`.

The cumulative explained variance is then calculated to determine how much of the total variance is retained as more principal components are included.

### 7. Selecting the Number of Components

The implementation selects the smallest number of principal components required to explain at least **95% of the total variance**.

```python
k = np.argmax(cumulative_variance >= 0.95) + 1
```

A plot of cumulative explained variance is also generated to visualize the relationship between the number of components and the amount of variance retained.

### 8. Constructing `U_k`

The first `k` eigenvectors are selected:

```python
U_k = eigenvectors[:, :k]
```

`U_k` contains the principal directions used for dimensionality reduction.

### 9. Projecting the Data

The standardized data is projected onto the selected principal directions:

```python
Z = X_std @ U_k
```

Mathematically:

\[
Z = X_{std}U_k
\]

where:

- `X_std` is the standardized original data
- `U_k` contains the selected principal directions
- `Z` is the reduced-dimensional representation of the data

## Output

The program displays:

- Dataset shape
- Features used
- Covariance matrix shape
- Eigenvalues
- Eigenvector shape
- Sorted eigenvalues
- Explained variance ratio
- Cumulative explained variance
- Number of selected principal components
- Shape of the original data
- Shape of the reduced data
- First five transformed samples

The program also generates a **cumulative explained variance plot**.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib

## What I Learned

Through this project, I practiced:

- Feature standardization
- Covariance matrices
- Eigenvalues and eigenvectors
- Orthogonal principal directions
- Explained variance
- Cumulative explained variance
- Projection into a lower-dimensional subspace
- Dimensionality reduction
- Matrix operations using NumPy

## Future Improvements

Possible improvements include:

- Visualizing the original and reduced data
- Comparing the implementation with Scikit-learn's PCA
- Experimenting with different variance thresholds
- Visualizing the first two principal components
- Applying PCA to other datasets

## Project Structure

```text
PCA/
│
├── principal component.py
├── student-mat.csv
└── README.md
```
