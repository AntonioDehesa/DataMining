# Vectors

Normally used to represent quantities that cannot be represented with just a number, but require a magnitude and a direction. 
But in CS, the term more commonly refers to a one-dimension array. Used to represent and manipulate data points. 

In data science, vectors can be used to represent either entire datasets, or one variable as one-hot encoding. 

They make it easy to perform mathematical operations on the variables in our dataset. They can also be easily transformed and modified. 

# Matrix

In math, it is a rectangular array of numbers, symbols or expressions, ordered in rows and columns. 
In Data science, it is used to represent and analyze datasets, when the data can be stored in a table format. 
They are regularly used in CS to represent observations, in which the first row represents the names of the columns, and the rest of the rows
represent a different observation. However, their uses are varied. 

They facilitate operations on the data, just as vectors. In fact, a matrix could be considered a vector of vectors. 

# Linear equations

Linear equations are those that can be represented simply by a variable or set of variables, multiplied by coefficients.
An example of this would be: a1x1 + a2b2 + a5b5 = y.
These coefficients can usually be considered parameters of the function. 

Often used in data science to model relationships between variables. The most common example of this would be the slope equation. 

An application of them would be for linear regression.

# Vector space in multiple dimensions

Set of V (elements of V are vectors), a field F (which can be scalars) and operations (vector addition and scalar product). 
In the case of multiple dimensions, I assume it means 3+ dimensions. In such case, to describe a single vector, you would need at least 3 scalars. 

In other words, it is a mathematical structure consiting of vectors (v) with specific rules and properties. 
Multiple dimensions would just mean that the vectors require at least one component in each of the referenced dimensions. That component can be of magnitude 0. 
In a vector space, vectors can be multiplied, added, substracted by scalars, following the rules previously mentioned. 

# Linear Transformations

Function that takes a vector and applies operations to turn it into a different vector, or to move them from one vector space to another. 

It needs to fulfill some conditions: 
* It preserves addition: If you add two vectors and apply the transformation, the result should be the same if you transform the vectors and
then you add them
* Same with products
We can imagine the vector "moving" in the dimension where it is. 
For it to be linear, the origin must not be changed, and lines must remain lines, not get distorted. 

It can be used for image processing, for example. Or for solving systems of linear equations. 

# Linear independence

Two vectors are independent from each other if there are no non-trivial linear combinations of the vectors that equal the zero vector. 
In this case, non-trivial means, for example, something other that multiplying by a zero vector.
If a vector can be represented as a linear combination of another vector, then that vector is not independent. 

Useful for determining if a feature/variable is redundant. If two variables are highly linearly dependent, then one of them provides redundant information, and could be omitted from the analysis for simplicity. 

# Eigenvalues and Eigenvectors

An Eigenvector is a vector which, after applying a linear transformation, only changes, at most, by a scalar factor, which is known as the Eigenvalue. 
The Eigenvalue may be positive or negative, but the general line of direction must remain the same for said vector.
By finding the Eigenvector, we know the axis of rotation of a 3D rotation, for example. This is because if the Eigenvector is the only vector that did not change other than scaling then every other vector changed around it. This is just an example. 

If the eigenvector is multiplied by it´s matrix, the output would be the same vector, multiplied by a scalar, which would be the eigenvalue. 

# Orthonormal bases and compliments

An Orthonormal Basis is a set of vectors thar are both ortogonal (perpendicular) and of length 1 (normal). They are used as the base of a system to describe other vectors, which are just a combination of the orthonormal vectors. 
For example, <3i,5j> would describe a vector that is 3 times the length of vector i in the direction of vector i, and 5 times the length of vector j, in the direction of j. 
Orthonormal compliments are a set of vectors that, when combined with another set of vectors, form an Orthonormal basis. 

# Compare and contrast data mining, machine learning and deep learning

* Data Mining: Field of CS to discover properties of a dataset, sorting through them to identify patterns and relationships that can help solve problems and make predictions. It uses multiple techniques, such as clustering, classification, association rule mining, etc. 
* Machine Learning: Sub-area of artificial intelligence. Technique to develop complex algorithms for processing large data, learning from it, and being able to provide predictions from it. It can be supervised or unsupervised. The algorithms usually learn patterns to make predictions. 
* Deep learning: Subset of machine learning that uses neural networks with 3 or more layers to learn from large datasets to be able to make predictions. 

The main similarity between these is that all of these techniques have the same goal: learn from a dataset to be able to create solutions, either by making predictions or by detecting relevant patterns and relationships. They also all require data preprocessing and feature engineering. 

Differences: 
data mining focuses on discovering properties of a dataset which can be used to make predictions, but can be used to learn more from the dataset and the business, while the other two focus on making predictions. 
machine learning is a wide field that can be used to make predictions through differente techniques
deep learning is a very specific subtechnique from machine learning that uses deep neural networks (at least 3 layers) to learn from a large dataset to make predictions

Data mining uses several different techniques to accomplish its goals, while the other two use a more focused approach. 