# Week 1

In Data Mining, unlike in Machine learning, we never split the data. 
Aspects of data mining: statistics, machine learning, pattern recognition, visualization, algorithms, high-performance computing, applications, information retrieval, data warehouse, etc. 

Data Silos: Companies used to split their data into different aspects of the company, and not even share it among the other departments of the same company. 
Data warehouse: You ETL the data from the silos and put it into the warehouse.
Data Lake: Same as the warehouse, but you do not ETL the data. It is in its raw state. This helps remove the bias and noise that could come from the ETL. 

# Week 2

## Itemset
items = elements

itemset = set of items

If all items are I, then X is a subset if all elements of X are in I. 
The cardinality (size) of the itemset is k.

## Tidset

Tids = transaction identifiers
Tidset = set of Tids

## Support and Frequem Itemsets

Support = Frequency of occurrence of an itemset. 

Minimum Support = The minimum occurrence of any itemset in a database. It can also be arbitrary, meaning that we can choose to ignore any itemset that does not fulfill this condition

An itemset is considered frequent if its support is at least equal to the minimum support. 

Relative support = The support divided by the total number of transactions

Confidence = Its the relation between two items and how confident we are that they are related. In other words, how confident we are that, if a person chooses item A, they will also choose item B. Defined by: confidence(A -> B): support(A U B) / Support(A)

## Algorithms for Itemset mining

### Brute force:
Get every combination of every item.
This technique is just not feasible, as it would demand enormous computational power. 

### Apriori Algorithm

Basically, we prune every branch of combinations that are not frequent enough. 
We assume that any superset of a not frequent set would also be not frequent, so we dont bother with creating it. 
More efficient than brute force, but still not very good

### Eclat algorithm

In this algorithm, instead of using the itemsets, we use the transactions.
We take transaction A and check the intersection with transaction B. 
We then check the support for the intersection. If its not at least equal to the minimum support, we stop the comparisons. We prune that branch.
Then we would take the intersection and check the intersection with another transaction. 

### DEclat algorithm

A variant of the previous algorithm. 
This one takes the difference instead of the intersection. 
Considered to be more efficient for small datasets, but worse for large datasets. 

### Frequent Pattern Tree (FPT) Algorithm

This algorithm starts with a null node. 
Then, an itemset is added to the null node, in order of most frequent to least frequent. 
Then, when another itemset is added, if it shares a path with the one already there, then a counter is added to the existing path setting it to two, and every other node to one. 
This lets us see the path that gets repeated the most, which would be the most frequent subset. 

### Generating Association Rules

First, we collect all frequent itemsets.
Any rule must fulfill the confidence and support requirements. 
Will write more about this when classes actually start

## Summarizing itemsets

### Maximal

An itemset is deemed maximal if it is frequent and has no frequent supersets. 

### Closed

An itemset is closed if all supersets of X have less support. 

## Minimal Generators

If a frequent itemset has no subsets with the same support, it is a minimal generator. 

It is not enough enough to simply determine that an itemset is frequent to mine the actual itemsets. Additional steps are required. 
Initially, the maximal itemset is empty. In order to add itemsets to the maximal frequent set, they must pass some tests: 
* item set y cannot be a member of the maximal set and that the item set x is a subset of item set y. If such a y item set exists, then the x item set cannot be maximal. If such an item set y does not exist, it is possible for item set x to be maximal.
* item set y cannot be a member of the maximal set and item set y cannot be a subset of item set x. If such an item set y exist, then y cannot be maximal. And if it is a member of the maximal set, it must be removed.

### GenMax algorithm

It works using tidsets. 
We set the tidsets for the elements (only those that fulfill the minimum support requirement)
Then those tidsets get intersected, and the intersection gets compared to the minimum support level. If it does not fulfill it, it gets pruned.
We do this with every combination until we get the maximal itemsets. 
Looks oddly similar to the Eclat algorithm. Because it is expandedd from it.

#### Differences: 
* Objectives: 
The Eclat algorithm wants to find all frequent itemsets
The GenMax algorithm just wants the maximal itemsets.

## Mining Closed Frequen Itemsets
CHARM algorithm.
Properties: 
* If t(x_i) == t(x_j) then c(x_i) == c(x_j) == c(x_i U x_j)
* If t(x_i)

will check later

# Week 3

## Non-derivable Itemsets

If the support of an itemset cannot be deduced from the support of its subsets, the itemset is non-derivable. 
The set of all frequent non-derivable itemsets represents all the frequent itemsets. --- what does this even mean? even chatgpt was confused.
does this mean that every frequent itemset is a non-derivable itemset?
The exact support for all frequent itemsets can be derived from the non-derivable itemset.

## Inclusion-Exclusion principle

Fundamental principle in combinatorial and set theories. 

It provides a way to calculate the size or cardinality of the union of multiple sets by considering the sizes of individual sets and their intersections.

This allows us to get the support level of non-derivable itemsets. 

### Example

sup(C not(AD)) = sup(C) - sup(AC) - sup(CD) + sup(ACD) = 4 -2 -2 +1 =1

### Lower and Upper Bounds

## Sequence Mining

* Sequential: Any data in which the sequence or order matters, such as time series
* Temporal: data that represents some state at an instant in time

* Lead nodes: nodes that have no children
* Internal nodes: Nodes that have children

Support of a sequence is the determinan of the S_i, if the sequence (r) belongs to s_i

As in sequences order matters, we can have infinite permutations in a database, so databases usually incorporate an upper bound length. 
### Generalized sequential patter mining

One approach is using sequence prefix trees to search level-wise or breadh first. 
Similar to previous methods. 

As it is too computationally expensive, we can use: 

### Spade algorithm

Uses vertical sequences. It is very similar to Eclat of FPT algorithms, as it uses the sequences, and checks in which ones it appears. 

### Projection based sequence mining

It kind of feels backwards. 
It starts by projecting all the sequences in the first node. 
Then, we select one element that interests us, or does not fulfill the minimum support, and we extract it (or several if a sequence) and mine the remaining sequences. 
We can conitnue doing this, recursively, which would improve the performance as the sequences are shorter and shorter. 
or so it seems. 
Need to ask in class

### Frequent substrings 

Identified by sequences that appear in leaf or internal nodes
Support of node labels must satisfy the assigned minsup
the prefixes of those leafs are also frequent

#### Ukkonen´s linear time algorithm

linear time online algorithm for constructing sufix trees

### Implicit suffixes 

If an extension is found in the tree during a phase, then all subsequent extensions will be found, so there is no need to process them 
These extensions are called implicit suffixes
The first time an extension is found, it is an explicit suffix

### Pattern and rule assessment 


# Week 4

* Clustering: Process of grouping similar items together in representative-based clusters
* Items are partitioned in K groups
* Dataset is described as n points in d-dimensional space
* Centroid: point that represents the summary of the cluster
* Stirling numbers of the second kind: Number of ways to partition a set of n objects into k non-empty subsets. 

## Representative-based clustering

Since any point can be assigned to any of the k clusters, it is possible to have O(k^n / k!) clusters
Instead of brute force, we can use either k-means or expectation-maximization algorithms.

### K-means

* Cluster assignment
* Centroid update
* Each point is assignmend to the closest mean
* Each point is assigned to cluster C_j

Once the new point is assigned, the mean of the cluster changes, as it is recalculated. 

### Kernel K-means

Allows for non-linear boundaries
Detects nonconvex clusters
Maps data points high-dimensional space using non-linear mapping
Allows for feature space to be explored by the function using dot product
Complexity: O(n^2) to compute the average kernel value for all clusters

### Expectation-Maximization Clustering

Algorithm that basically maximizes the probability (expectation) of an element belonging to a cluster. 
The sum of all the probabilities of an element belonging to each cluster must be equal to 1. 
Next part is thanks to chatgpt:
Here's a simplified explanation of the EM clustering algorithm:

Initialization:

Determine the number of clusters (K) you want to find.
Randomly assign each data point to one of the K clusters.
Expectation Step (E-step):

Calculate the probability of each data point belonging to each cluster.
Use the current cluster assignments and model parameters to estimate these probabilities.
This step is called the "expectation" step because it calculates the expected membership probabilities for each data point.
Maximization Step (M-step):

Update the model parameters based on the calculated probabilities in the E-step.
Optimize the cluster centroids (mean, median, or other representative points) based on the weighted contributions of the data points.
This step is called the "maximization" step because it maximizes the likelihood of the data by updating the model parameters.
Iteration:

Repeat the E-step and M-step iteratively until convergence.
In each iteration, the cluster assignments and model parameters are updated based on the data probabilities and optimized to improve the clustering.
Convergence:

The algorithm continues iterating until a stopping criterion is met, such as a maximum number of iterations or a negligible change in the cluster assignments.
Final Clustering:

Once the algorithm converges, the final cluster assignments are obtained based on the updated model parameters.
Each data point is assigned to the cluster with the highest probability.
The EM clustering algorithm is effective when dealing with data that might have overlapping or uncertain boundaries between clusters. It iteratively refines the cluster assignments and model parameters, gradually improving the clustering until convergence.

It's worth noting that the EM clustering algorithm assumes that the data follows a specific statistical distribution, often a Gaussian (normal) distribution. However, there are variations of the algorithm that can handle different types of data distributions.
There is also multiple dimension EM, but I will need a better example when the class happens.

K-means is a special instance of expectation maximization. 

### Maximum Likelihood Estimation

In order to get the maximum of something, we would get the first derivative of it and set it to 0. 
In the case of likelihood, its the same. For a set of parameters, we would get the partial derivative of every parameter and set them to 0. 
In this case, the parameters being the mean and the covariance matrix. 

## Hierarchical Clustering

* Clusters are a partition of the entire dataset. A cluster is considered to be trivial if it has every element in the dataset, or just a single element from the dataset. If a dataset has N elements, and there are N clusters, then every cluster will have a single element, so every cluster is trivial. 

* The hierarchy is built from the clusters that contain single points nested in the cluster that contains all points
* The dendrogram is a rooted binary tree that encompasses the nested structure
* Dendogram: nested partitions that can be visualized as a tree
* Agronin: An inverted dendogram
* leaves: nodes without branches
* internal nodes: nodes with branches
* The hierarchical tree will have some N nodes and N-1 of edges
* A rooted binary tree will have N number of leaves and N-1 internal nodes
* The total number of dendrograms that can be derived from n leaves is the product from m=1 to n-1 of (2m-1), which would be (2n-3)!

## Distance between clusters

* Euclidean Distance: Square root of the sum of the squared difference of every point. 
* Single Link: Minimum value of any of the x,y considerations where x is in the first cluster, and y is in the second cluster. 
* Complete Link: same as in single link, but instead of the minimum, its the maximum
* Group average: summation over both x and y. basically, cartesian product of all components in the x and y vectors, summed, and then divided by the product of the size of each


SSE: sum of squares = Sum from i = 1 to n of the predicted value minus the actual value, squared
For clusters, if is basicalle the same, but summed for every cluster

* Wards measure: difference between the SSEs for the clusters being compared = SSE_ij - SSE_i - SSE_j

Ward's measure, also known as Ward's linkage criterion, is a proximity measure used in hierarchical clustering to determine the dissimilarity between clusters during the merging process.
Ward's measure aims to minimize the increase in within-cluster variance when merging clusters. It takes into account the sum of squared differences within clusters and considers the distances between the cluster centroids.

The formula for Ward's measure is as follows:

Ward's measure = ∑(d^2) / (n - k)

Where:

∑(d^2) represents the sum of squared differences within clusters.
n is the total number of data points.
k is the current number of clusters being merged.

## Density-based Clustering

K-means and expectation-maximization methods are great for mining ellipsoid and convex clusters, but fail in non-convex space. 
In non-convex space it is possible to have two points in different clusters closer than two points in the same cluster.

### Density-based SCAN

SCAN stands for Spatial clustering of applications with noise

* DB clustering relies on the density of points in a cluster rather than the distance between the points
* DB clusters can be thought of as balls, the radius of the ball is referred to as E
* An E-neighborhood is defined as N_E(x) = B_d(X,E) = {y | delta(X,Y) =< E}
* Delta is a distance measure. Any will do
* A core point is any point in the E-neighborhood when minpts exist
* Minpts is user-defined density threshold
* If a point does not meet the minimum threshold of minpts, but it still belongs to the neighborhood, it is referred to as a border point
* A point is a noise point if it is not either a core or a border point
* A point is directly density reachable to another point if the other point is a core point
* A point is density reachable to another point if there are a set of core points between the points
* Two points are density connected if they are density reachable to a core point
* A set of density connected points form a density-based cluster

### Kernel Density Estimation

We can use a Kernel estimator to "smooth" over the data. 
Used to create a smooth curve given a set of data. 
Useful to visualize the shape of the data. It can also simulate points in a dataset. 
The function for the Kernel Density Estimation: 
(1/n*h)Sum from i=1 to n of K*((x-x_i)/h)
where K is the Kernel estimator

where: 
n = observations
K = estimator
h = bandwidthm which is the value we control
x = the point from which we are weighting the distances to our observations
#### Kernel estimator

* It must be non-negative at all points
* Must be symmetric around 0
* The total integral must equal 1, just like a CDF

#### Multivariate Density Estimation

The previous kernel method was univariate. 

We can create a hypersphere, like this: 
vol(H_d(h)) = h^d

f_hat(x) = 1/(n*h^d) sum of i=1 to n of K((x-x_i)/h)
where
integral of K(z)dz = 1
where
K(z) = 1 if |Z_j <= 1/2 for all dimensions j = 1,.....,d
0 otherwise
where z = x-x_i / h
and K(x-x_i/h) = 1

The Kernel Density Estimator, univariate, bivariate, or multivariate are not a direct method to perform clustering, but it is used by other methods that actually perform clustering

### Density Attractors and Gradient Ascent

Gradient Ascent: Opposite of Gradient Descent. Here we use the first derivative too, but in here, we get information to "go up" instead of down. 

* Density Attractor: This one points towards the densest cluster in the area, if certain criteria are met

Partial derivatives of f_hat(x) = i/n*h^d of sum from i = 1 to n of the partial derivatives for every dimension of K(x-x_i/h)
basically we get the vectors of the partial derivatives for every dimension, which then get added into a single vector

A density-based cluster is an arbitrary-shaped cluster that is an element of the entire dataset if density attractors exist such that: 
* The point is an element of the cluster
* The density of the density attractor is greater than xi
* The points on the path are density reachable to any of the density attractors

DENCLUE = Density-based Clustering

It is: 
* general case of kernel density estimate based clustering
* DBSCAN is a special case of kerne density estimate based clustering when where h = E and S = minpts
* The density attractors and the core points correspond
* The attractors are defined by a set of connected core points
* The significant cost of DENCLUE is the hill-climbing process
* Locating each attractor takes O(nt) where t is the iterations to climb the hill

### Cluster Validation

Techniques to confirm if the clustering techniques are correct or good.
Validation measures not inherent within the dataset are external measures
Those derived from the dataset are internal measures
Different clustering measures comparing cluster parameters are relative measures

#### External Measures

The ground-truth clustering is where all points in the cluster have the same label
Referring to the entire dataset is the ground-truth partition
Each subset of the dataset are referred to as partitions
External evaluation measure attempts to quantify which points from the same partition are included in the same cluster
They also qunaitify hot points from different partitions are grouped in different clusters
These evaluations can be quantified explicitly by measurements, or implicitly by computations

* Conditional Entropy: The lower the entropy, the better the organization
Entropy of the cluster = H(C) = - sum from i = 1 to r of P_C_i log(P_C_i)
Where P_C_i is the probability of the cluster. so it is the negative sum ot the probability of the cluster times the log of the probability. Log base 2

* Non normalized Hubert Discrete Statistic
= TP / N
measure of agreement or association between two categorical variables.
Ranges from -1 to 1
1 = perfect agreement
-1 = perfect disagreement

##### Matching Based Measure

Purity_cluster: Average of the maximum
Purity_dataset: Sum of the purity_cluster
precision and recall

#### Internal Measures

* They have no relationships with the ground-truth partitioning
* To evaluate the measures we must utilize intracluster similarities or compactness
* Internal measure of the pairwise distances are the distance matrix also referred to as the proximity matrix, or weights

????


### Cluster evaluation

Assesses the quality of clustering. 

### Cluster stability

Assesses sensitivity of the clusters

### Clustering Tendency

Assesses suitability of applying clustering techniques


# Week 6

## Linear methods of regression

* assume inputs are linear
* the regression function E(Y|X) provide an interpretable description of how inputs affect outputs
* often outperform more complicated models
* ideal for low signal to noise and sparse data

* RSS = Residual sum of squares, or sum of squared estimate of errors (SSE), is a measure of discrepancy between the data and an estimation model: = sum from i = 1 to n of (y_i - f(x_i))^2

### Hypothesis testing

Z_j = B_j / sigma*sqrt(v_j)

where B = is a parameter that can be estimated with the least squares method
B is the coefficient of the variables used in the estimator for the regression
if the null hypothesis is true, B_j = 0, as B_j has a minimum effect on the regression
a large Z_j infers a rejection of the null hypothesis
a known value of sigma then Z_j will have a normal distribution

