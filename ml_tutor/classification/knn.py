from ml_tutor.model import BaseModelClassification

from sklearn.decomposition import PCA
import random
import numpy as np
import matplotlib.pyplot as plt


class KNeighbourClassifier(BaseModelClassification):

	def __init__(self, n_neighbors=5, visual_training=True, number_of_visual_steps=-1):
		"""
		Create the K-Nearest Neighbors algorithm.

		:param n_neighbors: Number of neighbors to use when predicting a new data sample.
		:param visual_training: If True - the training process will be visualized [NOTE: only in Jupyter Notebook and Google Colab]
		:param number_of_visual_steps: Visualization might take a long time for this algorithm.
				With this parameter you can set how many points in the predictions time to visualize.
				NOTE: If you set this to -1 (by default) it will visualize ALL data samples you provide to it.
		"""
		super(BaseModelClassification, self).__init__()

		# Number of classes in the dataset
		# PCA is an unsupervised machine learning algorithm used to compress dataset and prepare it for visualization
		self.pca = PCA(n_components=2)
		self.predictions = None
		self.X_predict = None
		self.num_of_classes = None

		# Dataset -> X = features | y = labels/classes
		self.X = None
		self.y = None

		# Number of neighbours used to make predictions
		self.k = n_neighbors

		# Visualization related parameters
		self.visual_training = visual_training

		if not super().__is_visual_on__():
			self.visual_training = False
			print("Visualization is only supported in Jupyter Notebook and Google Colab.")

		self.number_of_visual_steps = number_of_visual_steps
		self.chars = '0123456789ABCDEF'
		self.colors = None

		# List that stores all distances
		self.distances = None

	def fit(self, X, y):
		"""
		Train the model using features (X) as training data and y as target values.

		:param X: Features from a dataset
		:param y: Target (classes) values (This is what you want to predict)
		"""
		self.X = X
		self.y = y

		if self.X.shape[1] > 2:
			if self.visual_training:
				print("The dataset is sparse for visual training. It will be pre-processed so it can be visualized.")
				print("Current shape of your data: {}".format(self.X.shape))
				self.X = self.pca.fit_transform(self.X)
				print("New shape of your data: {}".format(self.X.shape))

		self.num_of_classes = max(self.y) + 1
		self.colors = ['#' + ''.join(random.sample(self.chars, 6)) for i in range(self.num_of_classes)]

	def predict(self, X):
		"""
		This method performs predictions on the unseen data from your dataset.

		:param X: Data samples used to perform prediction on. (Generally a test set)
		:return: Predicted labels for each data sample
		"""

		if X.shape[1] > 2:
			if self.visual_training:
				X = self.pca.transform(X)

		self.X_predict = X
		self.predictions = []
		self.distances = []

		for i in range(len(X)):
			point = X[i]
			dist = np.linalg.norm(self.X - point, axis=1)
			self.distances.append(dist)
			idx = np.argpartition(dist, self.k)[:self.k]

			pred = np.argmax(np.bincount(self.y[idx]))
			self.predictions.append(pred)

		if self.visual_training:
			self.__visual_training__()

		return self.predictions

	def __visual_training__(self):
		"""
		Helper function used to crete real time visualization of the training process.
		"""
		# Import only relevant libraries for Jupyter Notebook if needed
		from IPython import display

		X = self.X_predict

		if self.number_of_visual_steps == -1:
			self.number_of_visual_steps = len(X)

		predicted_points = []

		for i in range(len(X)):
			point = X[i]
			dist = self.distances[i]
			if i < self.number_of_visual_steps:
				plt.close()
				plt.clf()
				plt.figure(figsize=(8, 6))
				plt.scatter(point[0], point[1], s=100)

				for k in range(self.num_of_classes):
					plt.scatter(self.X[self.y == k, 0], self.X[self.y == k, 1], c=self.colors[k], label="Class {} data samples".format(k))

				# Helper list for viz
				predicted_classes = []
				for p in range(len(predicted_points)):
					if self.predictions[p] not in predicted_classes:
						predicted_classes.append(self.predictions[p])
						plt.scatter(predicted_points[p][0], predicted_points[p][1], s=100,
						            c=self.colors[self.predictions[p]], label="Predicted points")
					else:
						plt.scatter(predicted_points[p][0], predicted_points[p][1], s=100,
						            c=self.colors[self.predictions[p]])

				for t in range(len(self.X)):
					current_x = self.X[t]

					rest_ids = []
					for j in range(len(self.X)):
						if j != i:
							rest_ids.append(j)

					distance_line = plt.plot([point[0], current_x[0]], [point[1], current_x[1]], "b", label="Distance line")

					m = [(point[k] + current_x[k]) / 2. for k in (0, 1)]
					text = plt.text(m[0], m[1], "%.2f" % dist[i])
					star = plt.scatter(current_x[0], current_x[1], c='black', marker="*")
					plt.title("KNeighbourClassifier - Prediction process")

					plt.legend(framealpha=1, frameon=True)
					display.display(plt.gcf())
					display.display()
					distance_line[0].remove()
					text.remove()
					star.remove()
					display.clear_output(wait=True)

				predicted_points.append(point)

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

		if super().__is_google_colab__():
			return "This method is not supported in Google Colab for now :/"

		contents = """
# If you don't have Sklearn installed execute line below
# pip install sklearn

# This is how you can import KNN using sklearn library
from sklearn.neighbors import KNeighborsClassifier

# Define classifier with selected parameters
model = KNeighborsClassifier(n_neighbors=5)

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
<h1>Machine Learning Basics with the K-Nearest Neighbors Algorithm</h1>
<br>
<br>

<p>

<h2>K-Nearest Neighbors</h2><br><br>
The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.
<br><br>
<img src="https://miro.medium.com/max/672/1*wW8O-0xVQUFhBGexx2B6hg.png">

<br><br>
Notice in the image above that most of the time, similar data points are close to each other. The KNN algorithm hinges on this assumption being true enough for the algorithm to be useful. KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics we might have learned in our childhood— calculating the distance between points on a graph.
<br><br>Note: An understanding of how we calculate the distance between points on a graph is necessary before moving on. If you are unfamiliar with or need a refresher on how this calculation is done, thoroughly read “Distance Between 2 Points” in its entirety, and come right back.
<br><br>There are other ways of calculating distance, and one way might be preferable depending on the problem we are solving. However, the straight-line distance (also called the Euclidean distance) is a popular and familiar choice.
<br><br>
<h2>The KNN Algorithm</h2> <br><br><br><br>
    1. Load the data <br><br>
    2. Initialize K to your chosen number of neighbors <br><br>
    3. For each example in the data <br><br>
        3.1 Calculate the distance between the query example and the current example from the data. <br><br>
        3.2 Add the distance and the index of the example to an ordered collection <br><br>
    4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances <br><br>
    5. Pick the first K entries from the sorted collection <br><br>
    6. Get the labels of the selected K entries <br><br>
    7. If regression, return the mean of the K labels <br><br>
    8. If classification, return the mode of the K labels <br><br>
<br><br>
<h2>Choosing the right value for K</h2> <br><br><br><br>
To select the K that’s right for your data, we run the KNN algorithm several times with different values of K and choose the K that reduces the number of errors we encounter while maintaining the algorithm’s ability to accurately make predictions when it’s given data it hasn’t seen before.
<br><br><br><br>Here are some things to keep in mind:
<br><br>    1. As we decrease the value of K to 1, our predictions become less stable. Just think for a minute, imagine K=1 and we have a query point surrounded by several reds and one green (I’m thinking about the top left corner of the colored plot above), but the green is the single nearest neighbor. Reasonably, we would think the query point is most likely red, but because K=1, KNN incorrectly predicts that the query point is green.
<br><br>    2. Inversely, as we increase the value of K, our predictions become more stable due to majority voting / averaging, and thus, more likely to make more accurate predictions (up to a certain point). Eventually, we begin to witness an increasing number of errors. It is at this point we know we have pushed the value of K too far.
<br><br>    3. In cases where we are taking a majority vote (e.g. picking the mode in a classification problem) among labels, we usually make K an odd number to have a tiebreaker.
<br><br><br><br><h2>Advantages</h2><br><br>
<br><br>    1. The algorithm is simple and easy to implement.
<br><br>    2. There’s no need to build a model, tune several parameters, or make additional assumptions.
<br><br>    3. The algorithm is versatile. It can be used for classification, regression, and search (as we will see in the next section).
<br><br><br><br><h2>Disadvantages</h2><br><br>
    1. The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.
</p>



<h1>Author and source:</h1>
<h2>Author: <a target="_blank" href="https://twitter.com/onelharrison">Onel Harrison</a></h2>
<h2>To find more resources go to the source of the post: <a target="_blank" href="https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761">Towards data science post</a></h2>

</div>
"""
			get_ipython().run_cell_magic(u'html', u'', content)

		else:
			content = u"""
<div>
<h1> K-Nearest Neighbors - How it works? </h1>
<iframe width="560" height="315" src="https://www.youtube.com/embed/44jq6ano5n0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
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
<h1> K-Nearest Neighbors Interview Questions </h1>

<h2> 1. What is “K” in KNN algorithm? </h2>
<p>
K = Number of nearest neighbors you want to select to predict the class of a given item
</p>

<h2> 2. How do we decide the value of "K" in KNN algorithm? </h2>
<p>
If K is small, then results might not be reliable because noise will have a higher influence on the result. If K is large, then there will be a lot of processing which may adversely impact the performance of the algorithm. So, following is must be considered while choosing the value of K:

a. K should be the square root of n (number of data points in training dataset)
b. K should be odd so that there are no ties. If square root is even, then add or subtract 1 to it.
</p>
<h2> 3. Why is the odd value of “K” preferable in KNN algorithm? </h2>
<p>
K should be odd so that there are no ties in the voting. If square root of number of data points is even, then add or subtract 1 to it to make it odd.
</p>
<h2> 4. What is the difference between Euclidean Distance and Manhattan distance? What is the formula of Euclidean distance and Manhattan distance? </h2>
<p>
Both are used to find out the distance between two points. <br><br>
  <img src="https://4.bp.blogspot.com/-9iDGWtgrbh0/XErhbogDiBI/AAAAAAAABLE/tAhILG2rJ68Hs88XBSi5PP0Wkxi3F-U2ACLcBGAs/s1600/Euclidean_distance_and_Manhattan_distance.PNG">
<br><br><br>
Euclidean Distance and Manhattan Distance Formula <br><br>
<img src="https://4.bp.blogspot.com/-Tr6BrJ4mZNw/XErlXGm8xrI/AAAAAAAABLc/ZWSmXXOrQBIyKmeO4DdPvckpUVjDFHW7wCLcBGAs/s1600/Euclidean_distance_and_Manhattan_distance_formula.PNG">
<br><br>
</p>
<h2> 5. Why is KNN algorithm called Lazy Learner? </h2>
<p>
When it gets the training data, it does not learn and make a model, it just stores the data. It does not derive any discriminative function from the training data. It uses the training data when it actually needs to do some prediction. So, KNN does not immediately learn a model, but delays the learning, that is why it is called lazy learner. 
</p>
<h2>  6. Why should we not use KNN algorithm for large datasets? </h2>
<p>
KNN works well with smaller dataset because it is a lazy learner. It needs to store all the data and then makes decision only at run time. It needs to calculate the distance of a given point with all other points. So if dataset is large, there will be a lot of processing which may adversely impact the performance of the algorithm. 

KNN is also very sensitive to noise in the dataset. If the dataset is large, there are chances of noise in the dataset which adversely affect the performance of KNN algorithm.	
</p>
<h3>  The questions and answers taken from: [<a href="http://theprofessionalspoint.blogspot.com/2019/01/knn-algorithm-in-machine-learning.html">link</a>]</h3>

<h3>  Quiz like questions: [<a href="https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-k-nearest-neighbors-algorithm/">link</a>] </h3>
"""
		get_ipython().run_cell_magic(u'html', u'', content)
