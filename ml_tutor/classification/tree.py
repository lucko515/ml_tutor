from ml_tutor.model import BaseModelClassification

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DecisionTreeClassification(BaseModelClassification):

	def __init__(self, criterion='gini', max_depth=None, visual_training=True, feature_names=None):
		"""
		Defines Decision Tree Classifier model.

		:param criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific. [From Sklearn]
		:param max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. [From Sklearn]
		:param visual_training: If True - the training process will be visualized [NOTE: only in Jupyter Notebook and Google Colab]
		:param feature_names: List of names for columns in a dataset. (List of strings)
		"""
		super(BaseModelClassification, self).__init__()

		# Number of classes in the dataset
		self.predictions = None
		self.criterion = criterion
		self.max_depth = max_depth

		# Dataset -> X = features | y = labels/classes
		self.X = None
		self.y = None

		# Visualization related parameters
		self.visual_training = visual_training
		if not super().__is_visual_on__():
			self.visual_training = False
			print("Visualization is only supported in Jupyter Notebook and Google Colab.")

		self.feature_names = feature_names

	def fit(self, X, y):
		"""
		Train the model using features (X) as training data and y as target values.

		:param X: Features from a dataset
		:param y: Target (classes) values (This is what you want to predict)
		"""
		self.X = X
		if isinstance(self.X, pd.DataFrame):
			self.feature_names = self.X.columns

		self.y = y

		from sklearn.tree import DecisionTreeClassifier

		self.classifier = DecisionTreeClassifier(criterion=self.criterion, 
												 max_depth=self.max_depth)

		self.figure = self.classifier.fit(self.X, self.y)

		if self.visual_training:
			self.__visual_training__()

	def predict(self, X):
		"""
		This method performs predictions on the unseen data from your dataset.

		:param X: Data samples used to perform prediction on. (Generally a test set)
		:return: Predicted labels for each data sample
		"""
		return self.classifier.predict(X)

	def __visual_training__(self):
		"""
		Helper function used to crete real time visualization of the training process.
		"""

		# Import only relevant libraries for Jupyter Notebook if needed
		from IPython import display
		from sklearn import tree
		plt.figure(figsize=(30, 20))
		tree.plot_tree(self.figure, feature_names=self.feature_names, filled=True, rounded=True)
		display.display(plt.gcf())
		display.display()
		display.clear_output(wait=True)

	def score(self, real, predicted):
		"""
		Return the accuracy computed on real vs. predicted classes.

		:param real: Expected targets(generally found in the dataset)
		:param predicted: Predicted classes by the algorithm

		:return: Mean accuracy computed on real vs. predicted classes [0. - 1.]
		"""
		assert len(real) == len(predicted)
		return sum(real == predicted) / len(real)

	def sklearn_version(self):
		"""
		Auto-generates sklearn code for a selected algorithm.

		NOTE: This function will automatically add one more code cell to your Jupyter Notebook/Google Colab (with the sklearn code inside).
		"""
		if not super().__is_visual_on__():
			print("Supported only in Jupyter Notebook and Google Colab.")
			return NotImplementedError

		if super().__is_google_colab__():
			return "This method is not supported in Google Colab for now :/"

		from IPython.core.getipython import get_ipython
		contents = """
# If you don't have Sklearn installed execute line below
# pip install sklearn

# This is how you can import DecisionTreeClassifier using sklearn library
from sklearn.tree import DecisionTreeClassifier

# Define classifier with selected parameters
model = DecisionTreeClassifier(criterion='gini', max_depth=None)

# Train the model using dataset you desire
model.fit(X_train, y_train)

# Finally, use trained model to make predictions
predictions = model.predict(X_test)

# Use Score method to make predictions
print(model.score(X_test, y_test))
"""
		shell = get_ipython()
		payload = dict(
			source='set_next_input',
			text=contents,
			replace=False,
		)
		shell.payload_manager.write_payload(payload, single=False)

	def how_it_works(self, video=False):
		"""
		Generates theory on how the algorithm works right in the Jupyter Notebook/Google colab.

		:param video: Some people prefer video tutorials over reading version. Set this parameter to True if you want video tutorial instead. :)
		"""
		if not super().__is_visual_on__():
			print("Supported only in Jupyter Notebook and Google Colab.")
			return NotImplementedError

		from IPython.core.getipython import get_ipython
		if not video:
			content = u"""
<div>
<h1>Decision Tree Algorithm — Explained</h1>
<img src="https://miro.medium.com/max/440/0*r8DWyN5pX4DRU89g.gif">
<br>
<br>

<p>
<h2>Introduction</h2>
<br><br>Classification is a two-step process, learning step and prediction step, in machine learning. In the learning step, the model is developed based on given training data. In the prediction step, the model is used to predict the response for given data. Decision Tree is one of the easiest and popular classification algorithms to understand and interpret.
<br><br><br><br>
<h2>Decision Tree Algorithm</h2><br><br>
Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.
<br><br>The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).
<br><br>In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

<br><br><br><br><h2>Types of Decision Trees</h2><br><br>
Types of decision trees are based on the type of target variable we have. It can be of two types:<br><br><br>
    Categorical Variable Decision Tree: Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.
    <br><br>Continuous Variable Decision Tree: Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree.
<br><br>


Example:- Let’s say we have a problem to predict whether a customer will pay his renewal premium with an insurance company (yes/ no). Here we know that the income of customers is a significant variable but the insurance company does not have income details for all customers. Now, as we know this is an important variable, then we can build a decision tree to predict customer income based on occupation, product, and various other variables. In this case, we are predicting values for the continuous variables.


<br><br><br><br><h2>Important Terminology related to Decision Trees</h2><br><br><br>
    Root Node: It represents the entire population or sample and this further gets divided into two or more homogeneous sets.
    <br><br>Splitting: It is a process of dividing a node into two or more sub-nodes.
    <br><br>Decision Node: When a sub-node splits into further sub-nodes, then it is called the decision node.
    <br><br>Leaf / Terminal Node: Nodes do not split is called Leaf or Terminal node.
    <br><br>Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say the opposite process of splitting.
    <br><br>Branch / Sub-Tree: A subsection of the entire tree is called branch or sub-tree.
    <br><br>Parent and Child Node: A node, which is divided into sub-nodes is called a parent node of sub-nodes whereas sub-nodes are the child of a parent node.
<br><br><br>
<img src="https://miro.medium.com/max/757/1*bcLAJfWN2GpVQNTVOCrrvw.png">
<br><br><br>Decision trees classify the examples by sorting them down the tree from the root to some leaf/terminal node, with the leaf/terminal node providing the classification of the example.
<br><br>Each node in the tree acts as a test case for some attribute, and each edge descending from the node corresponds to the possible answers to the test case. This process is recursive in nature and is repeated for every subtree rooted at the new node.


<br><br><br><br><h2>Assumptions while creating Decision Tree</h2><br><br><br>
    Below are some of the assumptions we make while using Decision tree:
    <br><br>In the beginning, the whole training set is considered as the root.
    <br><br>Feature values are preferred to be categorical. If the values are continuous then they are discretized prior to building the model.
    <br><br>Records are distributed recursively on the basis of attribute values.
    <br><br>Order to placing attributes as root or internal node of the tree is done by using some statistical approach.
<br><br><br>

Decision Trees follow Sum of Product (SOP) representation. The Sum of product (SOP) is also known as Disjunctive Normal Form. For a class, every branch from the root of the tree to a leaf node having the same class is conjunction (product) of values, different branches ending in that class form a disjunction (sum).
<br><br>
The primary challenge in the decision tree implementation is to identify which attributes do we need to consider as the root node and each level. Handling this is to know as the attributes selection. We have different attributes selection measures to identify the attribute which can be considered as the root note at each level.


<br><br><br><br><h2>How do Decision Trees work?</h2><br><br><br>
    The decision of making strategic splits heavily affects a tree’s accuracy. The decision criteria are different for classification and regression trees.
    <br><br>Decision trees use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that the purity of the node increases with respect to the target variable. The decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.
    <br><br>The algorithm selection is also based on the type of target variables. Let us look at some algorithms used in Decision Trees:
<br><br>ID3 → (extension of D3)
<br><br>C4.5 → (successor of ID3)
<br><br>CART → (Classification And Regression Tree)
<br><br>CHAID → (Chi-square automatic interaction detection Performs multi-level splits when computing classification trees)
<br><br>MARS → (multivariate adaptive regression splines)
<br><br><br>
The ID3 algorithm builds decision trees using a top-down greedy search approach through the space of possible branches with no backtracking. A greedy algorithm, as the name suggests, always makes the choice that seems to be the best at that moment.
<br><br><br><br>
Steps in ID3 algorithm:
<br><br>It begins with the original set S as the root node.
<br><br>On each iteration of the algorithm, it iterates through the very unused attribute of the set S and calculates Entropy(H) and Information gain(IG) of this attribute.
<br><br>It then selects the attribute which has the smallest Entropy or Largest Information gain.
<br><br>The set S is then split by the selected attribute to produce a subset of the data.
<br><br>The algorithm continues to recur on each subset, considering only attributes never selected before.

</p>



<h1>Author and source:</h1>
<h2>Author: <a target="_blank" href="https://twitter.com/nschauhan00">Nagesh Singh Chauhan</a></h2>
<h2>Read about overfitting and attributes in the source blog: <a target="_blank" href="https://towardsdatascience.com/decision-tree-algorithm-explained-83beb6e78ef4">Towards data science post</a></h2>

</div>
"""
			get_ipython().run_cell_magic(u'html', u'', content)
		else:
			content = u"""
<div>
<h1> Decision Tree - How it works? </h1>
<iframe width="560" height="315" src="https://www.youtube.com/embed/RmajweUFKvM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
"""
			get_ipython().run_cell_magic(u'markdown', u'', content)

	def interview_questions(self):
		"""
		Generates commonly asked interview questions about the algorithm in the Jupyter Notebook/Google colab.
		"""
		if not super().__is_visual_on__():
			print("Supported only in Jupyter Notebook and Google Colab.")
			return NotImplementedError

		from IPython.core.getipython import get_ipython

		content = u"""
<h1> Decision Tree Interview Questions </h1>

Quiz like questions: <a href="https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-tree-based-models/" target="_blank">link</a>	
"""
		get_ipython().run_cell_magic(u'html', u'', content)

