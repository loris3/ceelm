from apricot.optimizers import BaseOptimizer
from apricot.utils import PriorityQueue
import numpy
class LazyWeightedGreedy(BaseOptimizer):
	

	def __init__(self, function=None, random_state=None, n_jobs=None, 
		verbose=False, lambda_=0.9):
		super(LazyWeightedGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)
		self.lambda_ = lambda_

	def select(self, X, k, sample_cost=None):
		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		gains = self.function._calculate_gains(X) / sample_cost[self.function.idxs]
		self.pq = PriorityQueue(self.function.idxs, -gains)

		while len(self.function.ranking) < k:
			best_gain = float("-inf")
			best_idx = None
			
			while True:
				if len(self.pq.pq) == 0:
					return

				prev_gain, idx = self.pq.pop()
				#prev_gain, idx = self.pq.peek()
				prev_gain = -prev_gain

				# if cost + sample_cost[idx] > k:
				# 	continue

				if best_idx == idx:
					break
				
				idxs = numpy.array([idx])
				gain = self.function._calculate_gains(X, idxs)[0] / (sample_cost[idx]** self.lambda_)

				#self.pq.swap(idx, -gain)
				self.pq.add(idx, -gain)
				
				if gain > best_gain:
					best_gain = gain
					best_idx = idx
				elif gain == best_gain and best_gain == 0.0:
					best_gain = gain
					best_idx = idx
					break

			cost += sample_cost[best_idx]
			best_gain *= sample_cost[best_idx]
			self.function._select_next(X[best_idx], best_gain, best_idx)

			if self.verbose == True:
				self.function.pbar.update(1)
