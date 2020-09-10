from ml_tutor.model import BaseModelClassification

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


class LogisticRegression(BaseModelClassification):
	
	def __init__(self, learning_rate=0.1, num_iter=100000, fit_intercept=True, visual_training=True):
		"""
		Create Logistic Regression model.

		:param learning_rate: Learning rate gives the rate of speed where the gradient moves during gradient descent
		:param num_iter: Number of times to go through the dataset to train the model.
		:param fit_intercept: If True, bias will be used.
		:param visual_training: If True - the training process will be visualized [NOTE: only in Jupyter Notebook and Google Colab]
		"""
		super(BaseModelClassification, self).__init__()

		self.learning_rate = learning_rate
		self.num_iter = num_iter
		self.fit_intercept = fit_intercept
		
		self.visual_training = visual_training

		if not super().__is_visual_on__():
			self.visual_training = False
			print("Visualization is only supported in Jupyter Notebook and Google Colab.")

		self.pca = PCA(n_components=2)
		self.chars = '0123456789ABCDEF'
		self.colors = None
		
		self.theta = None
		self.theta_history = []
		self.threshold = 0.5
	
	def __add_intercept__(self, X):
		"""
		Add bias to the dataset
		:param X: Input data
		:return: Input data with concatenated column for bias
		"""
		return np.c_[np.ones((X.shape[0], 1)), X]
	
	def __sigmoid__(self, x):
		"""
		Sigmoid Activation function.
		:param x: Input data
		:return: Sigmoid values of the input data
		"""
		return 1 / (1 + np.exp(-x))

	def fit(self, X, y):
		"""
		Train the model using features (X) as training data and y as target values.

		:param X: Features from a dataset
		:param y: Target (classes) values (This is what you want to predict)
		"""
		from IPython import display
		
		self.X = X
		self.y = y
		
		if len(self.y.shape) < 2:
			self.y = np.expand_dims(self.y, axis=1)

		if len(self.X.shape) < 2:
			self.X = np.expand_dims(self.X, axis=1)
			
		self.num_of_classes = max(self.y)[0] + 1
		
		self.colors = ['#' + ''.join(random.sample(self.chars, 6)) for i in range(self.num_of_classes)]

		if self.num_of_classes > 2:
			message = "This algorithm supports ONLY binary classification. Max number of unique classes is two (2). \n Found: {} classes".format(self.num_of_classes)
			raise ValueError(message)
		
		if self.X.shape[1] > 2:
			if self.visual_training:
				print("The dataset is sparse for visual training. It will be pre-processed so it can be visualized.")
				print("Current shape of your data: {}".format(self.X.shape))
				self.X = self.pca.fit_transform(self.X)
				print("New shape of your data: {}".format(self.X.shape))
				
		if self.fit_intercept:
			self.X = self.__add_intercept__(self.X)
		
		# weights initialization
		self.theta = np.zeros((self.X.shape[1], 1))
		for i in range(self.num_iter):
			out = self.__sigmoid__(np.dot(self.X, self.theta))
			grad = np.dot(self.X.T, (out - self.y)) / self.y.size
			self.theta -= (self.learning_rate * grad)

			# NOTE: Visualization of the training process
			# TODO: Create better real time viz
			if self.visual_training:
				if i % 5000 == 0:
					# preds = self.visual_predict_prob(self.theta) >= self.threshold
			
					plt.close()
					plt.clf()
					plt.figure(figsize=(8, 6))

					for k in range(self.num_of_classes):
						plt.scatter(self.X[y == k, -2], self.X[y == k, -1], c=self.colors[k])

					x_values = [np.min(self.X[:, 1]) + 0.5, np.max(self.X[:, 2]) +1]
					y_values = - (self.theta[0] + np.dot(np.array(x_values).reshape(2, 1), self.theta[1].reshape(1, 1))) / self.theta[2]

					plt.plot(x_values, y_values, c='r', label="Classificatio line")
					display.display(plt.gcf())
					display.display()
					display.clear_output(wait=True)

	def predict_prob(self, X):
		"""
		This method performs predictions on the unseen data from your dataset but instead of return classes it returns probability of each class.

		:param X: Data samples used to perform prediction on. (Generally a test set)
		:return: Probability per class given data sample
		"""
		if self.fit_intercept:
			X = self.__add_intercept__(X)
	
		return self.__sigmoid__(np.dot(X, self.theta))
	
	def predict(self, X):
		"""
		This method performs predictions on the unseen data from your dataset.

		:param X: Data samples used to perform prediction on. (Generally a test set)
		:return: Predicted labels for each data sample
		"""
		return self.predict_prob(X) >= self.threshold

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
from sklearn.linear_model import LogisticRegression

# Define classifier with selected parameters
model = LogisticRegression()

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
# Logistic Regression

[TBA] Theory for Logistic Regression will be added here in a few days.		
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
# Logistic Regression Interview Questions

[TBA] Interview questions for Logistic Regression will be added here in a few days.		
"""
		get_ipython().run_cell_magic(u'markdown', u'', content)

