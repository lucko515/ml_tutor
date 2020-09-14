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

		if super().__is_google_colab__():
			return "This method is not supported in Google Colab for now :/"

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
<h1>Logistic Regression Explained</h1>
<br>
<img src="https://miro.medium.com/max/770/1*QY3CSyA4BzAU6sEPFwp9ZQ.png">
<br>
<br>
<p>

In the Machine Learning world, Logistic Regression is a kind of parametric classification model, despite having the word ‘regression’ in its name.<br><br>
This means that logistic regression models are models that have a certain fixed number of parameters that depend on the number of input features, and they output categorical prediction, like for example if a plant belongs to a certain species or not.

In Logistic Regression, we don’t directly fit a straight line to our data like in linear regression. Instead, we fit a S shaped curve, called Sigmoid, to our observations.
</p>
<br><br>
<img src="https://miro.medium.com/max/770/1*44qV8LhNzE5hPnta2PaaHw.png">
<br><br>
<p>
Let's examine this figure closely.
<br><br>First of all, like we said before, Logistic Regression models are classification models; specifically binary classification models (they can only be used to distinguish between 2 different categories — like if a person is obese or not given its weight, or if a house is big or small given its size). This means that our data has two kinds of observations (Category 1 and Category 2 observations) like we can observe in the figure.
<br><br>Note: This is a very simple example of Logistic Regression, in practice much harder problems can be solved using these models, using a wide range of features and not just a single one.
<br><br>Secondly, as we can see, the Y-axis goes from 0 to 1. This is because the sigmoid function always takes as maximum and minimum these two values, and this fits very well our goal of classifying samples in two different categories. By computing the sigmoid function of X (that is a weighted sum of the input features, just like in Linear Regression), we get a probability (between 0 and 1 obviously) of an observation belonging to one of the two categories.
<br><br>The formula for the sigmoid function is the following:
<br><br>
<img src="https://miro.medium.com/max/316/0*59BSXTBcxZcZGtVT">
<br><br>
<h2>1) Calculate weighted sum of inputs</h2>
<img src="https://miro.medium.com/max/271/0*vq7V-FuK9EirWDeN">
<br><br>
<h2>2) Calculate the probability of Obese</h2>
<img src="https://miro.medium.com/max/494/0*p5Yczl6itusXkxN8">
<br><br>
Alright, this looks cool and all, but isn’t this meant to be a Machine Learning model? How do we train it? That is a good question. There are multiple ways to train a Logistic Regression model (fit the S shaped line to our data). We can use an iterative optimisation algorithm like Gradient Descent to calculate the parameters of the model (the weights) or we can use probabilistic methods like Maximum likelihood.
</p>
<p>

<br><br>
Once we have used one of these methods to train our model, we are ready to make some predictions. Let's see an example of how the process of training a Logistic Regression model and using it to make predictions would go:
<br><br>1. First, we would collect a Dataset of patients who have and who have not been diagnosed as obese, along with their corresponding weights.

<br><br>2. After this, we would train our model, to fit our S shape line to the data and obtain the parameters of the model. After training using Maximum Likelihood, we got the following parameters:

<br><br>
<img src="https://miro.medium.com/max/487/1*sGI7P3PLzVcwWkLTE5Q7mg.png">

<br><br>
3. Now, we are ready to make some predictions: imagine we got two patients; one is 120 kg and one is 60 kg. Let's see what happens when we plug these numbers into the model:

<br><br>
<img src="https://miro.medium.com/max/520/1*fCwfXTnE55D_MJWNqhkegQ.png">


<br><br>
<img src="https://miro.medium.com/max/483/1*dwEPZB0Y6wKBvVNoQTLy6A.png">
</p>

<h1>Author and source:</h1>
<h2>Author: <a href="https://twitter.com/Jaimezorno">Jaime Zornoza</a></h2>
<h2>To find more resources go to the source post: <a href="https://towardsdatascience.com/logistic-regression-explained-9ee73cede081">Towards data science post</a></h2>

</div>
"""
			get_ipython().run_cell_magic(u'html', u'', content)
		else:
			content = u"""
<div>
<h1> Logistic Regression - How it works? </h1>
<iframe width="560" height="315" src="https://www.youtube.com/embed/-la3q9d7AKQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
"""
			get_ipython().run_cell_magic(u'html', u'', content)

	def interview_questions(self):
		"""
		Generates commonly asked interview questions about the algorithm in the Jupyter Notebook/Google colab.
		"""
		if not super().__is_visual_on__():
			print("Supported only in Jupyter Notebook and Google Colab.")
			return NotImplementedError

		from IPython.core.getipython import get_ipython

		content = u"""
<h1> Logistic Regression Interview Questions </h1>

Quiz like questions: <a href="https://www.analyticsvidhya.com/blog/2017/08/skilltest-logistic-regression/" target="_blank">link</a>
"""
		get_ipython().run_cell_magic(u'html', u'', content)

