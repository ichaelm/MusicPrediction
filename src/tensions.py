# Tension Module (numpy)
# Zicheng Gao
import numpy as np
import math

__EXPBASE = math.e
def softmax(x):
	return (__EXPBASE **x) / np.sum(__EXPBASE ** x)

v_basis = np.r_[:12]
z_12 = np.zeros(12)

_debug_ = False
__tritone__ = 6
_selfWeight_ = 0.6

valToNote = ['A','Bb','B','C','C#','D','Eb','E','F','F#','G','G#']

# Naive Tension Metric by interval
# tensions is a mapping of 0..12 to (0..6)^3
# tensions = [0,5,4,3,2,1,6,1,2,3,4,5]
# implement by operations (good with numpy)
# 0 1 2 3 4 5 6 7 8 9 A B
# 0 1 2 3 4 5 6 5 4 3 2 1
# 0 5 4 3 2 1 6 1 2 3 4 5

# Generate intervals between bases and quanta
def intervals(bases, quanta):
	diffs = np.tile(bases, (len(quanta), 1)) + 12
	diffs -= np.tile(quanta, (len(bases), 1)).T
	diffs %= 12
	return diffs

# Convert interval values to cubic tension values
# 11/15/16 - moved all up by one
def mut_intv_to_tension(intvs):
	lower = (intvs < 6) & (intvs > 0)
	intvs[lower] = 6 - intvs[lower]
	intvs[intvs > 6] -= 6
	intvs[intvs==6] = __tritone__
	tens = 3 ** (intvs-1)
	tens[intvs==0] = 0
	return tens + 1

def prepare_quanta(quanta):
	"""
	Ensure representation as numpy array and extract weights.
	If there are no weights, returns (notes, None)
	Otherwise returns (notes, weights)
	"""
	# ensure as numpy array
	if type(quanta) is not np.ndarray:
		quanta = np.array(quanta)

	if quanta.ndim > 1:
		return (quanta[:,0].astype(int), quanta[:,1])
	else:
		return (quanta.astype(int), None)


# evaluate sum of tensions between each self-note
def selfTension(notes_a):
	quanta, weights = prepare_quanta(notes_a)

	# for each note in the quanta determine the tension
	# symmetric matrix.
	diffs = intervals(quanta, quanta)
	tens = mut_intv_to_tension(diffs)

	noteamt = len(quanta)
	if weights is not None:
		tens = (tens.T * weights).T * weights
		noteamt = np.sum(weights)

	# print(tens)
	# maximum: all your nondiagonals are 243*3 (3**5)
	return np.sum(tens) / ((noteamt**2 - noteamt) * 3**4)

def inversionIntervals(quanta):
	qlen = len(quanta)
	intervals = np.tile(np.diff(quanta[np.r_[:qlen,0]])%12,qlen+1)
	return intervals.reshape(qlen,qlen+1)[:-1][:,:-1]

def inversionTensions(quanta):
	intervals = inversionIntervals(quanta)
	invTensions = np.sum(mut_intv_to_tension(intervals), 0)
	return np.vstack((quanta, invTensions)).T

# return energy for each basis note
# may be of use - but this should not truly determine basis (?)
def basis_likelihood(notes):
	quanta, weights = prepare_quanta(notes)

	if np.all(quanta == 0):
		return z_12 + 1/12

	diffs = intervals(v_basis, quanta)
	# tension matrix
	tens = mut_intv_to_tension(diffs)

	noteamt = len(quanta)
	if weights is not None:
		tens = tens * weights.T[:, np.newaxis]
		noteamt = np.sum(weights)

	tensionList = np.sum(tens, 0)

	# softmax(-log(kx)) = softmax(-log(x)), so no normalization or multiplication is needed
	return softmax(-np.log(tensionList))

def determineBasis(notes, tolerance=0, verbose=False):
	if verbose:
		print('for', notes)
	bTensions = np.vstack((v_basis, basis_likelihood(notes))).T
	base = np.max(bTensions[:,1])
	best = bTensions[bTensions[:,1] >= base * (1 - tolerance)]
	if verbose:
		for pair in best:
			print(valToNote[int(pair[0])] + ':', pair[1])
		print()
	return best

#################################################################
# Thoughts & possible future additions
# Maybe add some preference for lowest note in quanta as basis?
# Sort notes and then apply a scaling factor for notes that have many notes between them?

if __name__ == '__main__':

	if 1:
		# 'byInterval' can give a canonical answer but does not return other good candidates accurately (?)
		# Is simpler better sometimes?
		determineBasis([3, 7], verbose = True) # C major third => C / E
		determineBasis([3, 7, 10], verbose = True) # C MAJ => C w/o interval, G with
		determineBasis([3, 31, 34], verbose = True) # C MAJ across octaves => C
		determineBasis([27, 31, 34], verbose = True) # C MAJ one octave up => C
		determineBasis([3, 6,  9, 12], verbose = True) # C dim 7 = everything is bad
		determineBasis([3, 6, 10, 13], verbose = True) # C minor minor 7 => G / Eb
		determineBasis([3, 7, 10, 13], verbose = True) # C major minor 7 => G
		determineBasis([3, 6, 10, 14], verbose = True) # C major major 7 => G (!)

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
	# print(mut_intv_to_tension(np.array(v_basis)))
	print("Pairs")
	print(pairs)
	# print(selfTension(pairs))
	# print(determineBasis(np.array([0, 4, 7, 5]) + 3, tolerance = 1, verbose = True))

	print("Basis Likelihood")
	print(basis_likelihood(pairs))
	# determineBasis(pairs, verbose = True, tolerance = 1)

