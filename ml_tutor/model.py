from IPython.core.getipython import get_ipython

class BaseModelRegression(object):

	def fit(self, X, y):
		return NotImplementedError

	def predict(self, X):
		return NotImplementedError

	def __visual_training__(self):
		return NotImplementedError

	def score(self, real, predicted):
		return NotImplementedError

	def sklearn_version(self):
		return NotImplementedError

	def __is_visual_on__(self):
		try:
			shell = get_ipython().__class__.__name__
			if shell == 'ZMQInteractiveShell':
				return True  # Jupyter notebook or qtconsole
			elif shell == 'TerminalInteractiveShell':
				return False  # Terminal running IPython
			elif shell == 'Shell':
				return True  # Google Colab
			else:
				return False  # Other type (?)
		except NameError:
			return False

class BaseModelClassification(object):

	def fit(self, X, y):
		return NotImplementedError

	def predict(self, X):
		return NotImplementedError

	def __visual_training__(self):
		return NotImplementedError

	def score(self, real, predicted):
		return NotImplementedError

	def sklearn_version(self):
		return NotImplementedError

	def __is_visual_on__(self):
		try:
			shell = get_ipython().__class__.__name__
			if shell == 'ZMQInteractiveShell':
				return True  # Jupyter notebook or qtconsole
			elif shell == 'TerminalInteractiveShell':
				return False  # Terminal running IPython
			elif shell == 'Shell':
				return True  # Google Colab
			else:
				return False  # Other type (?)
		except NameError:
			return False


class BaseModelClustering(object):

	def fit(self, X):
		return NotImplementedError

	def predict(self, X):
		return NotImplementedError

	def __visual_training__(self):
		return NotImplementedError

	def sklearn_version(self):
		return NotImplementedError

	def __is_visual_on__(self):
		try:
			shell = get_ipython().__class__.__name__
			if shell == 'ZMQInteractiveShell':
				return True  # Jupyter notebook or qtconsole
			elif shell == 'TerminalInteractiveShell':
				return False  # Terminal running IPython
			elif shell == 'Shell':
				return True  # Google Colab
			else:
				return False  # Other type (?)
		except NameError:
			return False