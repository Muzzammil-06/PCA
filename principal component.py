import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("student-mat.csv", sep=";")

print("Dataset shape:", df.shape)
print(df.head())


# ============================================================
# 2. SELECT NUMERICAL FEATURES
# ============================================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

# Remove G3 because G3 is our target
X_df = numeric_df.drop(columns=["G3"])

print("\nFeatures used:")
print(X_df.columns)

X = X_df.to_numpy(dtype=float)

print("\nX shape:", X.shape)


# ============================================================
# 3. STANDARDIZE THE FEATURES
# ============================================================

# Calculate mean of each feature
mean = np.mean(X, axis=0)

# Calculate standard deviation of each feature
std = np.std(X, axis=0)

# Prevent division by zero
std[std == 0] = 1

# Standardize
X_std = (X - mean) / std

print("\nMean after standardization:")
print(np.mean(X_std, axis=0))

print("\nStandard deviation after standardization:")
print(np.std(X_std, axis=0))


# ============================================================
# 4. CALCULATE COVARIANCE MATRIX
# ============================================================

cov_matrix = np.cov(X_std, rowvar=False)

print("\nCovariance matrix shape:")
print(cov_matrix.shape)


# ============================================================
# 5. CALCULATE EIGENVALUES AND EIGENVECTORS
# ============================================================

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors shape:")
print(eigenvectors.shape)


# ============================================================
# 6. SORT EIGENVALUES FROM LARGEST TO SMALLEST
# ============================================================

# np.argsort gives indices that would sort the array
indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

print("\nSorted eigenvalues:")
print(eigenvalues)


# ============================================================
# 7. CALCULATE EXPLAINED VARIANCE
# ============================================================

explained_variance = (
    eigenvalues / np.sum(eigenvalues)
)

print("\nExplained variance ratio:")
print(explained_variance)


# ============================================================
# 8. CALCULATE CUMULATIVE EXPLAINED VARIANCE
# ============================================================

cumulative_variance = np.cumsum(explained_variance)

print("\nCumulative explained variance:")
print(cumulative_variance)


# ============================================================
# 9. PLOT EXPLAINED VARIANCE
# ============================================================

plt.figure()

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.title("PCA Explained Variance")

plt.grid()

plt.show()


# ============================================================
# 10. CHOOSE k
# ============================================================

# Keep enough components to explain at least 95% variance

k = np.argmax(cumulative_variance >= 0.95) + 1

print("\nNumber of components selected:", k)


# ============================================================
# 11. CREATE U_k
# ============================================================

U_k = eigenvectors[:, :k]

print("\nU_k shape:")
print(U_k.shape)


# ============================================================
# 12. PROJECT DATA INTO k DIMENSIONS
# ============================================================

Z = X_std @ U_k

print("\nOriginal data shape:")
print(X.shape)

print("\nReduced data shape:")
print(Z.shape)


# ============================================================
# 13. DISPLAY FIRST FEW REDUCED SAMPLES
# ============================================================

print("\nFirst 5 transformed samples:")

print(Z[:5])