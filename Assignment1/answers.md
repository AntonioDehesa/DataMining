# Vectors

Quantities that cannot be represented with just a number, but require a magnitude and a direction. 
But in CS, the term more commonly refers to a one-dimension array. 

# Matrix

In math, it is a rectangular array of numbers, symbols or expressions, ordered in rows and columns. 
But in CS, it can have more dimensions. 
They are regularly used in CS to represent observations, in which the first row represents the names of the columns, and the rest of the rows
represent a different observation. However, their uses are varied. 

# Linear equations

Linear equations are those that can be represented simply by a variable or set of variables, multiplied by coefficients.
An example of this would be: a1x1 + a2b2 + a5b5 = y.
These coefficients can usually be considered parameters of the function. 

# Vector space in multiple dimensions

Set of V (elements of V are vectors), a field F (which can be scalars) and operations (vector addition and scalar product). 
In the case of multiple dimensions, I assume it means 3+ dimensions. In such case, to describe a single vector, you would need at least 3 scalars. 

# Linear Transformations

Function that takes a vector and applies operations to turn it into a different vector. 
It needs to fulfill some conditions: 
* It preserves addition: If you add two vectors and apply the transformation, the result should be the same if you transform the vectors and
then you add them
* Same with products
We can imagine the vector "moving" in the dimension where it is. 
For it to be linear, the origin must not be changed, and lines must remain lines, not get distorted. 


# Linear independence

Two vectors are independent from each other if there are no non-trivial linear combinations of the vectors that equal the zero vector. 
In this case, non-trivial means, for example, something other that multiplying by a zero vector.
If a vector can be represented as a linear combination of another vector, then that vector is not independent. 

# Eigenvalues and Eigenvectors

An Eigenvector is a vector which, after applying a linear transformation, only changes, at most, by a scalar factor, which is known as the Eigenvalue. 
The Eigenvalue may be positive or negative, but the general line of direction must remain the same for said vector.
By finding the Eigenvector, we know the axis of rotation of a 3D rotation, for example. This is because if the Eigenvector is the only vector that did not change other than scaling then every other vector changed around it. This is just an example. 

# Orthonormal bases and compliments

An Orthonormal Basis is a set of vectors thar are both ortogonal (perpendicular) and of length 1 (normal). They are used as the base of a system to describe other vectors, which are just a combination of the orthonormal vectors. 
For example, <3i,5j> would describe a vector that is 3 times the length of vector i in the direction of vector i, and 5 times the length of vector j, in the direction of j. 
Orthonormal compliments are a set of vectors that, when combined with another set of vectors, form an Orthonormal basis. 

# Compare and contrast data mining, machine learning and deep learning

* Data Mining: Field of CS to discover properties of a dataset, dorting through them to identify patterns and relationships that can help solve problems and make predictions. 
* Machine Learning: Technique to develop complex algorithms for processing large data, learning from it, and being able to provide predictions from it. It can be supervised or unsupervised. 
* Deep learning: Subset of machine learning that uses neural networks with 3 or more layers to learn from large datasets to be able to make predictions. 

The main similarity between these is that all of these techniques have the same goal: learn from a dataset to be able to create solutions, either by making predictions or by detecting relevant patterns and relationships. 

Differences: 
data mining focuses on discovering properties of a dataset which can be used to make predictions, but can be used to learn more from the dataset and the business
machine learning is a wide field that can be used to make predictions through differente techniques
deep learning is a very specific subtechnique from machine learning that uses deep neural networks (at least 3 layers) to learn from a large dataset to make predictions