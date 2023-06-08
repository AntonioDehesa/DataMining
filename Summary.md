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