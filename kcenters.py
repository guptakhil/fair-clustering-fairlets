import numpy as np
import random
from utils import distance

class KCenters(object):

	def __init__(self, k=5):
		"""
		k (int) : Number of centers to be identified
		"""
		self.k = k

	def fit(self, data):
		"""
		Performs the k-centers algorithm.
	
		Args:
			data (list) : Points in the dataset
		"""
		# Randomly choosing an initial center
		#random.seed(42)

		self.data = data
		self.centers = [int(np.random.randint(0, len(self.data), 1))]
		self.costs = []
		
		while True:
			# Remaining points in the data set
			rem_points = list(set(range(0, len(self.data))) - set(self.centers))
			# Finding the point which has the closest center most far-off
			point_center = [(i, min([distance(self.data[i], self.data[j]) for j in self.centers])) for i in rem_points]
			point_center = sorted(point_center, key=lambda x: x[1], reverse=True)
			self.costs.append(point_center[0][1])
			if len(self.centers) < self.k:
				self.centers.append(point_center[0][0])
			else:
				break
		return

	def assign(self):
		"""
		Assigning every point in the dataset to the closest center.

		Returns:
			mapping (list) : tuples of the form (point, center)
		"""
		mapping = [(i, sorted([(j, distance(self.data[i], self.data[j])) for j in self.centers], key=lambda x: x[1], 
						   reverse=False)[0][0]) for i in range(len(self.data))]
		
		return mapping