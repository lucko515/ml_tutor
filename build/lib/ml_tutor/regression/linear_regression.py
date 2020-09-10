from ml_tutor.model import BaseModelRegression

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class LinearRegression(BaseModelRegression):

	def __init__(self, learning_rate=0.0001, num_iter=100000, tol=0.00001, visual_training=True):
		"""
		Creates the Linear Regression algorithm.

		:param learning_rate: Learning rate gives the rate of speed where the gradient moves during gradient descent
		:param num_iter: Number of times to go through the dataset to train the model.
		:param tol: If the difference between old and new values for model parameters are less than this number, training stops.
		:param visual_training: If True - the training process will be visualized [NOTE: only in Jupyter Notebook and Google Colab]
		"""
		super(BaseModelRegression, self).__init__()

		self.learning_rate = learning_rate
		self.num_iter = num_iter
		self.tol = tol
		self.visual_training = visual_training
		if not super().__is_visual_on__():
			self.visual_training = False
			print("Visualization is only supported in Jupyter Notebook and Google Colab.")

		self.randn_id = None

		# Gradient descent params
		self.starting_b = 0
		self.starting_m = 0
		self.b_history = []
		self.m_history = []

		print("If your dataset is sparse for visual training, random feature will be selected to match required shape.")
		print("Required shape for this algorithm is: [N, 1].")

	def fit(self, X, y):
		"""
		Train the model using features (X) as training data and y as target values.

		:param X: Features from a dataset
		:param y: Target values (This is what you want to predict)
		"""

		self.X = X
		self.y = y

		if len(self.y.shape) < 2:
			self.y = np.expand_dims(self.y, axis=1)

		if len(self.X.shape) < 2:
			self.X = np.expand_dims(self.X, axis=1)

		if self.X.shape[1] > 1:
			if self.visual_training:
				print("The dataset is sparse for visual training. This algorithm works only on shape [N, 1].")
				print("Random feature selected to match required size.")
				print("Current shape of your data: {}".format(self.X.shape))
				self.randn_id = np.random.randint(0, self.X.shape[1])
				print("Column selected on id: {}".format(self.randn_id))
				self.X = self.X[:, self.randn_id]
				if len(self.X.shape) < 2:
					self.X = np.expand_dims(self.X, axis=1)
				print("New shape of your data: {}".format(self.X.shape))

		# calling gradient descent function, and output of it is going to be our the best possible (according to our dataset) M and B
		self.__gradient_descent__(self.starting_b, self.starting_m)

	def __gradient_descent__(self, b, m):
		"""
		 main function for the gradient descent
		 :param b: Bias or constant
		 :param m: coefficient for X
		"""
		self.new_b = b
		self.new_m = m

		for i in range(self.num_iter):
			candidate_m, candidate_b = self.__gradient_descent_step__(self.new_b, self.new_m)

			if all(np.abs(candidate_m - self.new_m) <= self.tol) and \
					all(np.abs(candidate_b - self.new_b) <= self.tol):
				break

			self.new_m = candidate_m
			self.new_b = candidate_b

			if i % 1000 == 0:
				self.b_history.append(self.new_b)
				self.m_history.append(self.new_m)

		if self.visual_training:
			self.__visual_training__()

	def __visual_training__(self):
		"""
		Helper function used to crete real time visualization of the training process.
		"""

		# Import only relevant libraries for Jupyter Notebook if needed
		from IPython import display

		for i in range(len(self.b_history)):
			plt.close()
			plt.clf()
			plt.figure(figsize=(12, 10))

			plt.scatter(self.X, self.y, c='b', label="Training set")
			plt.plot(self.X, np.add(np.multiply(self.X, self.m_history[i]), self.b_history[i]), c='r',
			         label="Regression line")
			plt.title("Linear Regression - Training process")
			plt.xlabel("Feature value")
			plt.ylabel("Target value")
			plt.legend(framealpha=1, frameon=True)

			display.display(plt.gcf())
			display.display()
			time.sleep(1)
			display.clear_output(wait=True)

	def __gradient_descent_step__(self, b, m):
		"""
		Helper function for Gradient descent. Performs a single step of the gradient optimization.
		"""
		candidated_b = b - np.multiply(self.learning_rate,
		                               np.sum(-np.multiply(2 / float(len(self.X)),
		                                                   np.subtract(self.y,
		                                                               np.add(np.multiply(self.X, m), b))), axis=0))

		candidated_m = m - np.multiply(self.learning_rate,
		                               np.sum(np.multiply(2 / float(len(self.X)),
		                                                  np.multiply(-self.X,
		                                                              np.subtract(self.y,
		                                                                          np.add(np.multiply(self.X, m), b)))),
		                                      axis=0))

		return candidated_m, candidated_b

	def predict(self, X):
		"""
		This method performs predictions on the unseen data from your dataset.

		:param X: Data samples used to perform prediction on. (Generally a test set)
		:return: Predicted labels for each data sample
		"""
		if X.shape[1] > 2:
			if self.visual_training:
				X = X[:, self.randn_id]

		if len(X.shape[1]) < 2:
			X = np.expand_dims(X, axis=1)

		y_pred = np.add(np.multiply(X, self.new_m), self.new_b)

		return y_pred

	def score(self, real, predicted):
		"""
		Return the MSE computed on real vs. predicted classes.

		:param real: Expected targets(generally found in the dataset)
		:param predicted: Predicted values by the algorithm

		:return: Mean squared error computed on real vs. predicted classes [0. - 1.]
		"""
		assert len(real) == len(predicted)
		return mean_squared_error(real, predicted)

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

# This is how you can import LinearRegression using sklearn library
from sklearn.linear_model import LinearRegression

# Define regressor with selected parameters
model = LinearRegression()

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
# Linear Regression

[TBA] Theory for Linear Regression will be added here in a few days.		
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
# Linear Regression Interview Questions

[TBA] Interview questions for Logistic Regression will be added here in a few days.		
"""
		get_ipython().run_cell_magic(u'markdown', u'', content)
