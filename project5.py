#part A:
import numpy as np
from scipy.linalg import lu
from sklearn.decomposition import PCA


std_a=np.array([85,90,78])
std_b=np.array([76,82,68])

print("\nStudent std_a:", std_a)
print("\nStudent std_b:", std_b)

# Norm-1,2
print("\nNorm-1 ", np.linalg.norm(std_a, 1))
print("\nNorm-2 ", np.linalg.norm(std_a, 2))

# Dot product
dot = np.dot(std_a,std_b)
print("\nDot Product:", dot)

# Angle between vectors
angle = np.degrees(np.arccos(dot / (np.linalg.norm(std_a) * np.linalg.norm(std_b))))
print("\nAngle:",angle)
 
# Cross product
cross = np.cross(std_a, std_b)
print("\nCross Product:", cross)


# Projection 
projection = (np.dot(std_a, std_b) / np.dot(std_b,std_b)) * std_b
print("\nProjection :", projection)


#part B:

A = np.array([
    [80, 75, 90],
    [70, 85, 88],
    [95, 78, 92]
])

B = np.array([
    [5, 5, 5],
    [5, 5, 5],
    [5, 5, 5]
])

print("Matrix A:\n",A)

# Add
print("\nAddition:\n",A + B)

# Mul
print("\nMultiplication:\n",A @ B)

# Transpose
print("\nTranspose:\n",A.T)

# Inverse
print("\nInverse:\n",np.linalg.inv(A))

# Determinant
print("\nDeterminant:",np.linalg.det(A))

# part C:

# Plane
p_2D = np.array([4, 5])

# line
p_3D = np.array([4, 5, 6])

# Higher Dimension Hyperplane
p_4D = np.array([4, 5, 6, 7])

print("plane:",p_2D)
print("line:",p_3D)
print("Hyperplane:",p_4D)

# Dimensions
print("\n2D Dimension:",p_2D.ndim)
print("3D Dimension:",p_3D.ndim)
print("4D Dimension:",p_4D.ndim)

# part:D
#1#
A = np.array([[4, 2],
              [2, 3]])

values,vectors=np.linalg.eig(A)

print("Eigenvalues:\n",values)
print("Eigenvectors:\n",vectors)

#2#
A = np.array([[2, 1, 1],
              [4, -6, 0],
              [-2, 7, 2]])

# LU 
P, L, U = lu(A)

print("L Matrix:\n",L)
print("U Matrix:\n",U)

#3#
A=np.array([[1, 2],
              [3, 4],
              [5, 6]])

# SVD
U,S,VT= np.linalg.svd(A)

print("U Matrix:\n",U)
print("Singular Values:\n",S)
print("VT Matrix:\n",VT)

# part:E

# PCA

A=np.array([
    [80, 75, 90],
    [70, 85, 88],
    [60, 78, 95],
    [90, 92, 85]
])

pca=PCA(n_components=2)

reduced=pca.fit_transform(A)

print("Reduced Data:",reduced)


#LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Student marks
X = np.array([
    [80, 75],
    [70, 85],
    [60, 78],
    [90, 92]
])

# Categories
# 1 = Above Average
# 0 = Below Average
y = np.array([1, 0, 0, 1])

lda = LinearDiscriminantAnalysis(n_components=1)

result = lda.fit_transform(X, y)
print("result:",result)



