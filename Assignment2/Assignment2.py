# Imports
from sklearn import datasets
import numpy as np
import pandas as pd
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
print(np.linalg.norm(sepal_length))

print("Distance between the sepal length vector and the sepal width vector")
print(np.linalg.norm(sepal_length - sepal_width))