---
published: true
layout: post
title: 'Summary of Chapter 6, Decision Trees'
---

What is Decision Tree? How is it different from other models? Let's understand our first non-linear model, non-parametric model. 何为决策树？跟其他模型区别在哪儿？来了解我们第一个非线性的非参数的模型。

*In this chapter summary, I will briefly introduce how Decision Tree in Scikit-Learn works, as well as measurement of impurity.*
*在这节总结中，我将展示决策树的简单原理.*

#### Table of Content 内容提要  

* CART    
* Impurity  
* Cost function  
* Probability  
* Regularization   
* Regression  

Like SVMs (introduced in Chapter 5 Summary), Decision Trees are versatile Machine Learning algorithms that can perform both classification and regression tasks. They are very powerful algorithms, capable of fitting complex datasets.

### What is a Decision Tree

What is a decision tree? Decision Trees are tree-like models, where each node contain a subset of the datasets and, if it is a nonleaf node, makes decisions on how the subset can be further split into more branches.  
!['Decision Tree'](../images/handson/chap6_iris_tree.png)  
In the chart above, there are 150 samples in the root node (top-level node). These samples are split into two subsets based on whether their *pental length<=2.45*. 

### Classification and Regression Tree (CART)
There are many algorithms for Decision Trees. In Scikit-Learn uses the CART algorithm, which produces only binary trees: nonleaf nodes always have two children. As you can tell from the name, CART can be applied to both classification and regression problems. You can probably tell that there are also algorithms produces more than just two children; one example is ID3. 
 
### Impurity 
At nonleaf nodes, Decision Trees make decision on how to further split the dataset based on certain criteria. How are these criteria decided? Intuitively,  in a classification problem, we wish a split could separate instances of different classes as distinct as possible. In the best case, one subset will contain instances from one class, the other subset the other class. In another word, we hope the subset as *pure* as possible. And the best split should give the least impurity.  
Now we need some measurements of impurity. Here are two common ones.  

* __gini_impurity__  
!['gini_impurity'](../images/handson/chap6_gini_impurity.png) 

* __Entropy__  
!['Entropy'](../images/handson/chap6_entropy.png) 

subscript $i$ is for nodes, subscript $k$ for classes. For example, in the decision tree shown just now, at the root node, $p_{i,1}=p_{i,2}=p_{i,3}=\frac{50}{150}=0.33$. $G_i=1-\sum_{k=1}^3p_{i,k}^2=1-3*0.33*0.33=0.667$. 

It is easy to tell that these two metric changes almost the same way when $p_{i,k}$ changes. So they are almost equivalent.  

A reduction of entropy is often called an **information gain**.

### Cost function for Classification 
To minimize the impurity, we can subsequently define the cost function of each split as 
!['Cost function for classification'](../images/handson/chap6_cost_function_classification.png) 

### Regularization
Now we know how to split at a single node. Iterating this process on all the produced children node generates a decision tree. 

When do we stop splitting? If we only stop when there is no more impurity, we are obviously just remembering the mapping, which is an extreme case of overfitting. To regularize, there are various hyperparameters to give the decision tree growing an early stop. Common hyperparameters are `max_depth`, `min_sample_split`, etc. And again, cross-validation is our good friend on checking overfitting.

### Probability 
A Decision Tree can also estimate the probability that an instance belongs to a particular class k: first it traverses the tree to find the leaf node for this instance, and then it returns the ratio of training instances of class k in this node.

### Regression 
We have previously mentioned that CART can also be used for classification. The CART algorithm works mostly the same way as earlier, except that instead of trying to split the training set in a way that minimizes impurity, it now tries to split the training set in a way that minimizes the MSE.

!['Cost function for regression'](../images/handson/chap6_cost_function_regression.png) 

#### Discussion
A single decision tree model suffers a few issues. For example, the split can only be orthogonal to the axes, which makes it suffer from dataset rotation. Besides, Decision Trees are sensitive to small variations in the training data.  
How to overcome these issues? You will see it in the next summary, the Ensemble methods. 


*This article is part of a series of summaries on the book Hands-On Machine Learning with Scikit-Learn and TensorFlow. The summaries are meant to explain machine learning concepts and ideas, instead of covering the maths behind the models.* 

*本文是《Hands-On Machine Learning with Scikit-Learn and TensorFlow》这本书的总结随笔系列的一部分。总结旨在解释机器学习的观念和想法，而不是数学和模型*