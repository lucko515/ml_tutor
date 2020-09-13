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
		"""
		if not super().__is_visual_on__():
			print("Supported only in Jupyter Notebook and Google Colab.")
			return NotImplementedError


		from IPython.core.getipython import get_ipython


		if not video:
			content = u'''
<h1> K-Nearest Neighbors - How it works? </h1>
# NOTE: Temporary holder

KNN is a non-parametric and lazy learning algorithm. Non-parametric means there is no assumption for underlying data distribution. In other words, the model structure determined from the dataset. This will be very helpful in practice where most of the real world datasets do not follow mathematical theoretical assumptions. Lazy algorithm means it does not need any training data points for model generation. All training data used in the testing phase. This makes training faster and testing phase slower and costlier. Costly testing phase means time and memory. In the worst case, KNN needs more time to scan all data points and scanning all data points will require more memory for storing training data.

## How does the KNN algorithm work?

In KNN, K is the number of nearest neighbors. The number of neighbors is the core deciding factor. K is generally an odd number if the number of classes is 2. When K=1, then the algorithm is known as the nearest neighbor algorithm. This is the simplest case. Suppose P1 is the point, for which label needs to predict. First, you find the one closest point to P1 and then the label of the nearest point assigned to P1.

![](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/Knn_k1_z96jba.png)

Suppose P1 is the point, for which label needs to predict. First, you find the k closest point to P1 and then classify points by majority vote of its k neighbors. Each object votes for their class and the class with the most votes is taken as the prediction. For finding closest similar points, you find the distance between points using distance measures such as Euclidean distance, Hamming distance, Manhattan distance and Minkowski distance. KNN has the following basic steps:

  - Calculate distance
  - Find closest neighbors
  - Vote for labels
  

![](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final1_ibdm8a.png)

## How do you decide the number of neighbors in KNN?

Now, you understand the KNN algorithm working mechanism. At this point, the question arises that How to choose the optimal number of neighbors? And what are its effects on the classifier? The number of neighbors(K) in KNN is a hyperparameter that you need choose at the time of model building. You can think of K as a controlling variable for the prediction model.

Research has shown that no optimal number of neighbors suits all kind of data sets. Each dataset has it's own requirements. In the case of a small number of neighbors, the noise will have a higher influence on the result, and a large number of neighbors make it computationally expensive. Research has also shown that a small amount of neighbors are most flexible fit which will have low bias but high variance and a large number of neighbors will have a smoother decision boundary which means lower variance but higher bias.

Generally, Data scientists choose as an odd number if the number of classes is even. You can also check by generating the model on different values of k and check their performance. You can also try Elbow method here.

![](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final_a1mrv9.png)


# To learn more about KNN go to DataCamp post [here](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=278443377086&utm_targetid=aud-390929969673:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1028595&gclid=Cj0KCQjw-af6BRC5ARIsAALPIlXXK_ItCNKM3FkFQpSH3oBIPB0Wm5cSs43HCt_qYyjAE8CPqGfUynAaAhYSEALw_wcB)

## Source for text and images is DataCamp post.		
		
'''
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

		if super().__is_google_colab__():
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
Both are used to find out the distance between two points. 
  <img src="https://4.bp.blogspot.com/-9iDGWtgrbh0/XErhbogDiBI/AAAAAAAABLE/tAhILG2rJ68Hs88XBSi5PP0Wkxi3F-U2ACLcBGAs/s1600/Euclidean_distance_and_Manhattan_distance.PNG">
  
Euclidean Distance and Manhattan Distance Formula
<img src="https://4.bp.blogspot.com/-Tr6BrJ4mZNw/XErlXGm8xrI/AAAAAAAABLc/ZWSmXXOrQBIyKmeO4DdPvckpUVjDFHW7wCLcBGAs/s1600/Euclidean_distance_and_Manhattan_distance_formula.PNG">
(Image taken from stackexchange)
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
		else:
			content = u"""
# K-Nearest Neighbors Interview Questions

## 1. What is “K” in KNN algorithm?

K = Number of nearest neighbors you want to select to predict the class of a given item

## 2. How do we decide the value of "K" in KNN algorithm?

If K is small, then results might not be reliable because noise will have a higher influence on the result. If K is large, then there will be a lot of processing which may adversely impact the performance of the algorithm. So, following is must be considered while choosing the value of K:

a. K should be the square root of n (number of data points in training dataset)
b. K should be odd so that there are no ties. If square root is even, then add or subtract 1 to it.

## 3. Why is the odd value of “K” preferable in KNN algorithm?

K should be odd so that there are no ties in the voting. If square root of number of data points is even, then add or subtract 1 to it to make it odd.

## 4. What is the difference between Euclidean Distance and Manhattan distance? What is the formula of Euclidean distance and Manhattan distance?

Both are used to find out the distance between two points. 
![](https://4.bp.blogspot.com/-9iDGWtgrbh0/XErhbogDiBI/AAAAAAAABLE/tAhILG2rJ68Hs88XBSi5PP0Wkxi3F-U2ACLcBGAs/s1600/Euclidean_distance_and_Manhattan_distance.PNG)
Euclidean Distance and Manhattan Distance Formula
![](https://4.bp.blogspot.com/-Tr6BrJ4mZNw/XErlXGm8xrI/AAAAAAAABLc/ZWSmXXOrQBIyKmeO4DdPvckpUVjDFHW7wCLcBGAs/s1600/Euclidean_distance_and_Manhattan_distance_formula.PNG)
(Image taken from stackexchange)

## 5. Why is KNN algorithm called Lazy Learner?

When it gets the training data, it does not learn and make a model, it just stores the data. It does not derive any discriminative function from the training data. It uses the training data when it actually needs to do some prediction. So, KNN does not immediately learn a model, but delays the learning, that is why it is called lazy learner. 

## 6. Why should we not use KNN algorithm for large datasets?

KNN works well with smaller dataset because it is a lazy learner. It needs to store all the data and then makes decision only at run time. It needs to calculate the distance of a given point with all other points. So if dataset is large, there will be a lot of processing which may adversely impact the performance of the algorithm. 

KNN is also very sensitive to noise in the dataset. If the dataset is large, there are chances of noise in the dataset which adversely affect the performance of KNN algorithm.	

### The questions and answers taken from: [link](http://theprofessionalspoint.blogspot.com/2019/01/knn-algorithm-in-machine-learning.html)

### Quiz like questions: [link](https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-k-nearest-neighbors-algorithm/)
"""
			get_ipython().run_cell_magic(u'markdown', u'', content)
