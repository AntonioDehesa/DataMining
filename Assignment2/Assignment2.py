# Intent of the application

#The purpose of this program is to explore the Iris dataset as the first programming assignment for the Data Mining course, perform exploratory operations, display features of the dataset, and create plots for the dataset.

# Dataset to be used, including source

#The dataset to be used is the Iris dataset, for which more information can be found here: https://www.ritchieng.com/machine-learning-iris-dataset/

# Use case

#The use case of this application could be to demonstrate basic operations with datasets, as the first steps to take when exploring a dataset for the first time. 

# Variables

#Sepal.Length: The length of the sepal, which is the outer part of the flower that encloses a developing bud. 
#Sepal.Width: The width of the sepal.
#Petal.Length: The length of the petal of the flower, which are leaves that surround the reproductive parts of a flower. 
#Petal.Width: The width of the petal of the flower. 

# Labels

#Species: Describes the species of the flower associated to the previous measurements. 

# Data import

#In this application, there is no input needed from the user.

# Libraries

#Sklearn: Used to import the Iris dataset. 
#Numpy: Used to obtain the norm of a vector, and the distance between two vectors
#Pandas: Used to manage the vectors in the Iris dataset, as well as to perform the dot product between two vectors.

# Imports
from sklearn import datasets
from numpy.linalg import norm
import pandas as pd
from matplotlib import pyplot as plt

# Library source

#Sklearn: Source and latest changes -> https://scikit-learn.org/dev/whats_new/v1.3.html
#Numpy: Source and latest releases -> https://numpy.org/news/
#Pandas: Source and latest changes -> http://pandas.pydata.org/pandas-docs/stable/

# Application outputs

#The expected outputs for this application are: 

#Dot product between the sepal length and the sepal width vectors
#Norm of the sepal length vector
#Distance between the sepal length and the sepal width vectors

# Proposed solution

# Now we import the dataset
iris = datasets.load_iris()

# We can perform a similar exploration as we did in R
print("Feature names (predictor columns)")
print(iris["feature_names"])

print("Possible outcomes")
print(iris["target_names"])

# Now we calculate the dot product between the sepal length vector and the sepal width vector
iris_df = pd.DataFrame(data = iris["data"], columns = iris["feature_names"])
print("First five values of each predictor column")
print(iris_df.head())


print("Dot product between sepal length and sepal width")
sepal_length = iris_df["sepal length (cm)"]
sepal_width = iris_df["sepal width (cm)"]
print(sepal_length@sepal_width)

print("Norm of the sepal length vector")
print(norm(sepal_length))

print("Distance between the sepal length vector and the sepal width vector")
print(norm(sepal_length - sepal_width))


# Analysis of results

#Unfortunately, the dot product by itself does not tell us much. In this case, we can only see that the dot product is possitive, and it has a high magnitude. This could indicate a high degree of correlation, but it is not guaranteed. 

#With the norm of the sepal length vector, we can tell that the magnitudes of each point in the vector may not be very large. 
#Finally, with the distance obtained between the sepal length and the sepal width vector we can see that it was low, which would confirm, with the large dot product previously obtained, that the two vectors are highly related. 


# Visualizations
fig, ax = plt.subplots()

ax.plot(sepal_length, sepal_width, linewidth=2.0)
plt.show()