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

	def how_it_works(self):
		"""
		Generates theory on how the algorithm works right in the Jupyter Notebook/Google colab.
		"""
		if not super().__is_visual_on__():
			print("Supported only in Jupyter Notebook and Google Colab.")
			return NotImplementedError

		from IPython.core.getipython import get_ipython

		content = u"""
# Decision Tree Classifier

[TBA] Theory for Decision Tree Classifier will be added here in a few days.	
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
# Decision Tree Classifier Interview Questions

[TBA] Interview questions for Decision Tree Classifier will be added here in a few days.		
"""
		get_ipython().run_cell_magic(u'markdown', u'', content)
