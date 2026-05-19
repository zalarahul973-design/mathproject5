# Linear Algebra & Machine Learning Concepts in Python

This project demonstrates various **Linear Algebra operations**, **matrix computations**, and **basic Machine Learning techniques** using Python libraries such as NumPy and Scikit-learn.

---

## 📌 Project Overview

The code is divided into multiple parts:

- **Part A:** Vector operations (norm, dot product, angle, cross product, projection)
- **Part B:** Matrix operations (addition, multiplication, transpose, inverse, determinant)
- **Part C:** Dimensional analysis (2D, 3D, 4D representations)
- **Part D:** Advanced linear algebra (Eigenvalues, LU decomposition, SVD)
- **Part E:** Machine Learning techniques (PCA and LDA)

---

## 🧮 Part A - Vector Operations

### Features:
- L1 Norm and L2 Norm
- Dot Product
- Angle between vectors
- Cross Product
- Projection of one vector onto another

### Libraries Used:
- `numpy`

---

## 🧮 Part B - Matrix Operations

### Features:
- Matrix Addition
- Matrix Multiplication
- Transpose
- Inverse
- Determinant

### Note:
Matrix inverse is calculated using `np.linalg.inv()`.

---

## 📐 Part C - Dimensional Representation

### Features:
- Representation of:
  - 2D vector (Plane)
  - 3D vector (Line/Space)
  - 4D vector (Hyperplane)
- Checking dimensions using `.ndim`

---

## 🔢 Part D - Advanced Linear Algebra

### 1. Eigenvalues & Eigenvectors
- Used for understanding transformations

### 2. LU Decomposition
- Matrix decomposition into Lower (L) and Upper (U) matrices using `scipy.linalg.lu`

### 3. Singular Value Decomposition (SVD)
- Factorization of matrix into U, Σ, and Vᵀ

### Libraries Used:
- `numpy`
- `scipy.linalg`

---

## 🤖 Part E - Machine Learning Concepts

### 1. Principal Component Analysis (PCA)
- Reduces dataset dimensions while preserving variance
- Implemented using `sklearn.decomposition.PCA`

### 2. Linear Discriminant Analysis (LDA)
- Supervised classification technique
- Used for separating classes based on features

### Libraries Used:
- `sklearn.decomposition`
- `sklearn.discriminant_analysis`



