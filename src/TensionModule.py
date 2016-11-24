# Tension Module (numpy)
# Zicheng Gao

import numpy as np

v_basis = np.r_[:12]

valToNote = ['A','Bb','B','C','C#','D','Eb','E','F','F#','G','G#']

# from Robert Rowe's "Machine Musicianship" p 48
class metric:
	dissonance = [1, 6, 5, 4, 3, 2, 7, 2, 3, 4, 5, 6]
	western = [1, 8, 6, 2.1, 2, 4.5, 5, 1.5, 4.6, 4.7, 4, 7]

class TensionModule:

	def __init__(self, metric = metric.dissonance, kertosis = 1):
		self.metric = metric
		self.kertosis = kertosis

	def mut_tens(self, intvs):
		# change to tensions
		tens = np.tile(intvs, (1,1))
		for i in np.r_[:12]:
			tens[intvs == (12 - i) % 12] = self.metric[i]
		return 3 ** tens

	@staticmethod
	def __intervals(bases, quanta):
		# Generate intervals between bases and quanta
		diffs = np.tile(bases, (len(quanta), 1)) + 12
		diffs -= np.tile(quanta, (len(bases), 1)).T
		diffs %= 12
		return diffs

	@staticmethod
	def __prepare_quanta(quanta):
		# Extract Pairs of Notes and Weights into numpy arrays.
		# Return (notes, weights). With no weights, weights returns None.
		if type(quanta) is not np.ndarray:
			quanta = np.array(quanta)

		if quanta.ndim > 1:
			return (quanta[:,0].astype(int), quanta[:,1])
		else:
			return (quanta.astype(int), None)

	def selfTension(self, notes_a):
		# evaluate sum of tensions between each self-note
		quanta, weights = self.__prepare_quanta(notes_a)
		diffs = self.__intervals(quanta, quanta)
		tens = self.mut_tens(diffs)
		noteamt = len(quanta)
		if weights is not None:
			tens = (tens.T * weights).T * weights
			noteamt = np.sum(weights)
		return np.sum(tens) / ((noteamt**2 - noteamt) * 3**4)

	def inversionIntervals(quanta):
		# return intervals of the inversions of this note group
		qlen = len(quanta)
		intervals = np.tile(np.diff(quanta[np.r_[:qlen,0]])%12,qlen+1)
		return intervals.reshape(qlen,qlen+1)[:-1][:,:-1]

	def inversionTensions(self, quanta):
		# return tensions of inversions based on internal intervals
		intervals = self.inversionIntervals(quanta)
		invTensions = np.sum(self.mut_tens(intervals), 0)
		return np.vstack((quanta, invTensions)).T

	def basis_likelihood(self, notes):
		# return likelihood based off minimal energy for each basis note
		# probably shouldn't be used to actually determine basis
		quanta, weights = self.__prepare_quanta(notes)
		if np.all(quanta == 0):
			return np.zeros(12) + 1/12
		diffs = self.__intervals(v_basis, quanta)
		tens = self.mut_tens(diffs)
		noteamt = len(quanta)
		if weights is not None:
			tens = tens * weights.T[:, np.newaxis]
			noteamt = np.sum(weights)
		tensionList = np.sum(tens, 0)
		return self.flipmax(tensionList)

	def determineBasis(self, notes, tolerance=0, verbose=False):
		# print out as well - has internal trim
		if verbose:
			print('for', notes)
		bTensions = np.vstack((v_basis, self.basis_likelihood(notes))).T
		base = np.max(bTensions[:,1])
		best = bTensions[bTensions[:,1] >= base * (1 - tolerance)]
		if verbose:
			for pair in best:
				print(valToNote[int(pair[0])] + ':', pair[1])
			print()
		return best

	# the recently made-up cousin of softmax
	def flipmax(self, x):
		return (1 / (x ** self.kertosis)) / np.sum(1 / (x ** self.kertosis))


#################################################################
# Thoughts & possible future additions
# Maybe add some preference for lowest note in quanta as basis?
# Sort notes and then apply a scaling factor for notes that have many notes between them?

if __name__ == '__main__':

	tensmod = TensionModule()

	if 1:
		# 'byInterval' can give a canonical answer but does not return other good candidates accurately (?)
		# Is simpler better sometimes?
		tensmod.determineBasis([3, 7], verbose = True) # C major third => C / E
		tensmod.determineBasis([3, 7, 10], verbose = True) # C MAJ => C w/o interval, G with
		tensmod.determineBasis([3, 31, 34], verbose = True) # C MAJ across octaves => C
		tensmod.determineBasis([27, 31, 34], verbose = True) # C MAJ one octave up => C
		tensmod.determineBasis([3, 6,  9, 12], verbose = True) # C dim 7 = everything is bad
		tensmod.determineBasis([3, 6, 10, 13], verbose = True) # C minor minor 7 => G / Eb
		tensmod.determineBasis([3, 7, 10, 13], verbose = True) # C major minor 7 => G
		tensmod.determineBasis([3, 6, 10, 14], verbose = True) # C major major 7 => G (!)

	if 0:
		print("==Tensions==\n")

		# Thirds
		print('Major Third', selfTension([0, 4]))
		# print(selfTension([2, 6]))

		# Fifths
		print('Perfc Fifth', selfTension([2, 9]), '\n')
		# print(selfTension([2, 9, 2, 9]))

		print('Augmented Triad', selfTension([0, 4, 8]))
		print('Major Triad', selfTension([0, 4, 7]))
		print('Minor Triad', selfTension([0, 3, 7]))
		print('Diminished Triad', selfTension([3, 7, 9]), '\n')

		print('Fully Dimin 7', selfTension([0, 3, 6, 9])) # should be very high
		print('Minor Minor 7', selfTension([0, 3, 7, 10])) # should be ok
		print('Minor Major 7', selfTension([0, 3, 7, 11])) # should be high
		print('Dominant    7', selfTension([0, 4, 7, 10])) # should be low, but the tritone between the 3 and the 7- makes this bad
		print('Major Major 7', selfTension([0, 4, 7, 11])) # should be ok

		print()
		print('Cluster', selfTension([0, 1, 2, 3]))
		print('Cluster', selfTension([0, 1, 2]))
		
	keys = np.r_[:88]
	m_state = np.zeros(88)
	m_state[0] = 1.0
	m_state[12] = 1.0
	m_state[12 + 3] = 1.0
	m_state[12 + 7] = 1.0
	m_state[24 + 3] = 1.0
	# m_state[7 + 24] = 1.0
	# m_state[0:88] = 1.0
	# length = 1+1*6
	# m_state[0:length:6] = 1.0

	actives = keys[m_state > 0]
	weights = m_state[m_state > 0]

	pairs = np.vstack((actives, weights)).T
	# print(mut_tens(np.array(v_basis)))
	print("Pairs")
	print(pairs)
	# print(selfTension(pairs))
	# print(determineBasis(np.array([0, 4, 7, 5]) + 3, tolerance = 1, verbose = True))

	print("Basis Likelihood")
	print(tensmod.basis_likelihood(pairs))
	# determineBasis(pairs, verbose = True, tolerance = 1)

