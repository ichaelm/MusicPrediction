"""
Based on:
Vos, Piet G., and Erwin W. Van Geenen.
“A Parallel-Processing Key-Finding Model.”
Music Perception: An Interdisciplinary Journal,
vol. 14, no. 2, 1996, pp. 185–223.
www.jstor.org/stable/40285717.

Zicheng Gao
"""
import numpy as np

__empty = np.zeros((2, 12)) + (1 / (2*12))

# Metric of contributions, not tensions
class metric:
	# The contributions of A to the scales of <index> pitch
	# We don't care about mode - just go to ionian, always
	scalar = np.tile(np.array([
		[1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], # major
		[1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0], # harmonic minor
		# [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], # natural minor
		]), (1, 2))

	# same, to chords
	chordal = np.tile(np.array([
		[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], # major
		[1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], # harmonic minor
		# [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], # natural minor
		]), (1, 2))

# TODO: Make some sort of utility class?
def __prepare_quanta(quanta):
	# Extract Pairs of Notes and Weights into numpy arrays.
	# Return (notes, weights). With no weights, weights returns None.
	if type(quanta) is not np.ndarray:
		quanta = np.array(quanta)
	if quanta.ndim > 1:
		return (quanta[:,0].astype(int), quanta[:,1])
	else:
		return (quanta.astype(int), None)

def note_score(note, metric):
	note %= 12
	return metric[..., 12-note:24-note]

def group_score(group, metric):
	(quanta, weights) = __prepare_quanta(group)

	length = np.size(quanta, 0)

	if length is 0:
		return __empty

	output = np.zeros((2, 12))

	# move weight presence check up for performance (?)
	if weights is not None:
		for i in np.r_[:length]:
			output += weights[i] * note_score(quanta[i], metric)
	else:
		for i in np.r_[:length]:
			output += note_score(quanta[i], metric)

	output = output / np.sum(output)
	output = output * (output == np.max(output))

	return output

def get_key(group):
	output = group_score(group, metric.scalar) + group_score(group, metric.chordal)
	output = output / np.sum(output)
	return output

if __name__ == '__main__':
	section = [0, 2, 4, 5, 7]
	# print(group_score(section, metric.chordal))
	print(get_key(section))
	# print(group_score([1, 5, 8], metric.scalar))