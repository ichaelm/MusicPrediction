# Attribute basis to MRL - (do not individually apply to MTP)

# http://www.music.mcgill.ca/~jason/mumt621/papers5/fujishima_1999.pdf ?

# Zicheng (Brian) Gao

from MusicRoll import *
import tensions
import numpy as np
import pickle
import pprint

from tkinter import Tk, Button, Frame, Canvas, Scrollbar
import tkinter.constants as Tkconstants

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

pp = pprint.PrettyPrinter(indent=4)

# roll = pickle.load(open('./mid/mary.mrl', 'rb'))
# roll = pickle.load(open('./mid/oldyuanxian2.mrl', 'rb'))
# roll = pickle.load(open('./mid/start_offset.mrl', 'rb'))
roll = pickle.load(open('./mid/ambigious_test.mrl', 'rb')) # lowsim 0.3 - 0.35
# roll = pickle.load(open('./mid/ivivi.mrl', 'rb')) # lowsim 0.5
# roll = pickle.load(open('./mid/two_channel_test.mrl', 'rb'))
# roll = pickle.load(open('./mid/channel_sep.mrl', 'rb'))

pp.pprint(vars(roll))

time = 0
attn_decay = 0.3 # decay BY this much
w_lmin = 4 # minimum window size for labelling
w_emin = 4 # maximum hiccup size - for dealing with neighbour tones and _small_ interruptions in basis

# threshold for ambiguosity - for the most likely basis, candidates must be above ambg * most_likely_likelihood
# if there are multiple candidates, it is an ambiguous section
ambg = 0.5

# similarity threshold for crossing hiccups
# higher is more sensitive
low_sim = 0.3

# match parameters
# match_bias = 0.5
# penalty for lowering existing belief
# match_pdrop = 1
# penalty for alternate belief
# match_palt = 0.5

def ema(seq, inertia):
	# exponential moving average
	for i in np.r_[1:np.size(seq, 0)]:
		seq[i] = inertia * seq[i-1] + (1 - inertia) * seq[i]
	return seq

def trim(bvec, push = 1.0):
	# push 0 -> bvec
	# push 1 -> 1
	# factor = (bvec + push * (1 - bvec)) = b + p - bp = b (1 - p) + p
	t = 1.0 * (bvec > (ambg * np.max(bvec)))
	if push == 1.0:
		return t
	else:
		return (bvec * (1 - push) + push) * t

def matches(belief, actual):
	# see if the measured is allowed by the belief
	# that which was high in belief should not decrease by too much

	# match lowered by:
	# actual is lower on existing belief
	# p_match = min(match_pdrop * np.sum((actual - belief)[trim(belief) == 1]), 0)
	# actual is higher on alternate belief
	# p_drop = match_palt * np.sum(actual[trim(belief) == 0])

	# similarity = match_bias - p_drop - p_match
	# print(similarity)
	# return similarity
	return np.dot(belief, actual)


# for label in roll.labels:
	# print("Label")
	# pp.pprint(vars(label))

for tempo, group in roll.get_tape_groups().items():

	print(group)
	# for tape in group:
		# pp.pprint(vars(tape))

	# Array - newness matters, as novelty strikes the ears - simulate attentional decay
	note_data = MusicRoll.combine_notes(group)

	# 0 -> 1-k
	# 1 -> 1
	# factor = (1-k) + n * k
	# 		 = 1 - k + nk = 1 - k * (n - 1)
	notes = note_data[...,0] * (1 - attn_decay * (note_data[...,1] - 1) )
	orig_notes = notes

	# smooth notes somewhat
	notes = ema(np.tile(notes, 1), 0.25)

	duration = np.size(notes, 0)
	keys = np.r_[:np.size(notes, 1)]

	fig1 = plt.figure(1, figsize = (5, 5))
	
	grid = AxesGrid(fig1, 111, 
		nrows_ncols = (3, 1),
		axes_pad = 0.05,
		label_mode = "1",
		)

	

	grid[2].locator_params(axis='y', nbins = 12)

	basis_l = np.zeros((duration, 12))#likelihoods
	basis_b = np.zeros((duration, 1)) - 1#labels
	tension = np.zeros(duration)

	def axis_basis(quanta):
		actives = keys[quanta > 0.0001] + 3 # due to midi pitch nonsense - 0 is C
		weights = quanta[quanta > 0.0001]
		return tensions.basis_likelihood(np.vstack((actives, weights)).T)

	def label(base, start, end):
		if start == 0:
			start -= 1
		grid[2].broken_barh([(start + 1, end - start)], (base + 0.25 , 0.5), facecolors = 'red', alpha = 0.3)
		basis_b[start:end] = base

	def bar(time):
		grid[2].broken_barh([(time, 0.1)], (0, 12), facecolors = 'red', alpha = 0.7)

	def block(base, time):
		grid[2].broken_barh([(time, 0.1)], (base + 0.25, 0.5), facecolors = 'blue', alpha = 0.7)
		
	# go through the time slices...
	W = [[0, 0]]
	E = [[0, 0]]

	# candidacy
	candidate = -1

	def label_basis():
		# print("LBL", b_curr, W[0][0])
		for base in tensions.v_basis[b_curr == np.max(b_curr)]:
			label(base, W[0][0], W[0][1])


	while W[0][1] < duration:
		# big window
		if W[0][1] - W[0][0] >= w_lmin:
			# get current candidates
			section = np.sum(notes[W[0][0]:W[0][1]+1], 0)
			basis_l[W[0][1]] = b_curr = axis_basis(section)
			section = np.sum(orig_notes[W[0][0]:W[0][1]+1], 0)
			tension[W[0][1]] = tensions.selfTension(section)
			# print("CUR", cand_curr, W[0][1])

			# check the next part to see if it is a subset of current candidates, or doesn't match
			E[0][0] = 1
			similarity = 0
			while E[0][0] < w_emin and W[0][1] + E[0][0] < duration and similarity < low_sim:
				b_next = axis_basis(notes[W[0][1] + E[0][0]])
				# print("NXT", cand_next, W[0][1] + E[0][0])
				# print(tensions.v_basis[trim(b_curr) > ambg])
				similarity = matches(trim(b_curr, 1), b_next)
				# print("SHR", similarity)

				if similarity < low_sim:
					E[0][0] += 1
			
			# now that we are out of that, see if we exceeded the window - otherwise it was similar so we can proceed as usual
			# became dissimilar
			if similarity < low_sim:
				# does the beginning of the window agree with the end?
				# (problem of the cooking frog - alternate candidate increased slowly, displacing the original candidate(s))

				bar(W[0][1] + 1)
				# label_basis()
				W[0][0] = W[0][1]
		# small window
		else:
			basis_l[W[0][1]] = b_curr = axis_basis(notes[W[0][1]])
			tension[W[0][1]] = tensions.selfTension(np.sum(orig_notes[W[0][0]:W[0][1]+1], 0))
			# TODO also label here?

		W[0][1] += 1

		# back-label

	# label when hitting end
	# label_basis()

	"""
	It's like this.
	Doing one run doesn't really cut it - we have to do another run to actually label.
	Mostly, due to the impact that ambg has on the actual "smoothing" and "inference."

	There are several approaches to this 1D Sliding Window Detection

		Window identification and merging
			This is a granular approach but should be pretty reliable for large-scale identification.
			This will probably misbehave for highly varied bases.
			< L/W * Log2(L/W) >
		Top-down window division sliding
			Probably inefficient
			< L^2 / W >
		Candidacy abandoning w/ forward-pushing window, bridging gaps
			Is a gap supposed to be bridged?
			Jump a gap instead?
				Label across gap, or...?
			< 2L (probe all gaps, bridge as much as possible ) >

	In all approaches, confidence should be adjusted by number of notes available
		This info can be gotten through summing "newness" across the section in the window.
		Probably:
			multiply by < 1 - (1/notes) >
	"""
	
	#####################
	### labelling run ###
	#####################

	# forward-pushing, bridging gaps

	# singular candidate for basis
	l_cand = -1
	# cutting threshold for settling on a candidate
	l_ambg = 0.5
	# drop percentage for abandoning candidate (and perform label)
	l_chng = 0.2
	# granularity. the size at which it's no longer a gap / minimum label width
	l_gaps = 4

	lleft = 0 # start of section to label - for bridging gaps
	left = 0  # start of section to examine
	right = 0 # 
	while right < duration:
		# while smaller than window
		while right - lleft < l_gaps:
			# attempt to find candidate
			

		# tcandidate, tconfidence = factor * best @ right
		# update best confidence if more confident

		# HAVE CANDIDATE
		# if best match prev and confidence didn't drop over chng and confidence is good enough:
			# keep going
		# DON'T HAVE CANDIDATE
		# else:
			# start an exploration here at 'E'
				# get best @ E
					# if best match this match
						# if past window?
							

		right += 1

	# basis_l = np.apply_along_axis(axis_basis, 1, notes)

	grid[0].set_title("{0} group {1}".format(roll.filepath, tempo))
	min_note = min([tape.min_note for tape in group])
	max_note = max([tape.max_note for tape in group])

	grid[0].plot(np.r_[:duration] + 0.5, 100 * tension, 'k')
	grid[1].imshow(notes.T[min_note:max_note + 1],
		interpolation = 'none',
		cmap = plt.cm.Oranges,
		origin = 'lower',
		extent=[0, duration, 0, 12],
		aspect = 0.5 * duration / 12)
	grid[2].imshow(basis_l.T,
		interpolation = 'none', 
		cmap = plt.cm.Greys,
		origin = 'lower',
		extent=[0, duration, 0, 12],
		aspect = 0.5 * duration / 24)

	# print(basis_b)

	plt.show()


# This is really a classification problem that ought to be addressed with the proper tools



# get it to mark what actions it took - exploring, backlabelling
# prevent "consecutive self-labelling" for example
# and also smooth away hiccups