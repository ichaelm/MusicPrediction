# Attribute basis to MRL - (do not individually apply to MTP)

# http://www.music.mcgill.ca/~jason/mumt621/papers5/fujishima_1999.pdf ?
# https://www.jstor.org/stable/pdf/40285717.pdf

# Zicheng (Brian) Gao

from MusicRoll import *
import TensionModule
import PPMBasis
import numpy as np
import pickle
import pprint

from tkinter import Tk, Button, Frame, Canvas, Scrollbar
import tkinter.constants as Tkconstants

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

__verbose = False
__block = True
__report_interval = 100
__hard_limit = 1000

# parameters:

attn_decay = 0.3 # decay by this much from note onsets
n_smoothing = 0.0 # forward smoothing on notes

# candidates must be above b_ambiguous * most_likely_likelihood
# if there are multiple candidates, it is an ambiguous section
# lesser cuts more; higher makes ambiguous sections more likely
# thus, higher is actually more stringent
b_ambiguous = 0.4
max_gap = 4
thresh_conf = 0.25
# lower bound for continuation of current hypothesis
persistence = 0.25

def debug_print(*args):
	if __verbose:
		print(args)
		
def ema(seq, inertia):
	# mutating operation: exponential moving average applied over an array
	for i in np.r_[1:np.size(seq, 0)]:
		seq[i] = inertia * seq[i-1] + (1 - inertia) * seq[i]
	return seq

def apply_with_window(origin, target, vfunction, width, wdirection = 'backwards'):
	# wdirection must either be 'forwards' or 'backwards'
	assert np.size(origin, 0) == np.size(target, 0)

	# branching for efficiency? save calculations in loop?
	if wdirection == 'backwards':
		for i in np.r_[0:width]:
			target[i] = vfunction(origin[ 0 : i ])
		for i in np.r_[width:np.size(origin, 0)]:
			target[i] = vfunction(origin[ i - width : i ])


def trim(vector, factor):
	return 1.0 * (vector > ((1 - factor) * np.max(vector)))

def matches(belief, actual):
	# How much does the observed value match the hypothesis
	return 1 - np.linalg.norm(belief - actual)

def do_basis_label(filename, metric = TensionModule.metric.dissonance):
	pp = pprint.PrettyPrinter(indent=4)
	roll = pickle.load(open(filename, 'rb'))
	pp.pprint(vars(roll))

	tens_mod = TensionModule.TensionModule(metric)

	for tempo, group in roll.get_tape_groups().items():
		max_gap = 2 * max([tape.min_common for tape in group])

		note_data = MusicRoll.combine_notes(group)

		# Array - newness matters - simulate attentional decay
		# @newness = 0 -> 1-k
		# @newness = 1 -> 1
		# factor = (1-k) + n * k
		# 		 = 1 - k + nk = 1 - k * (n - 1)
		orig_notes = note_data[...,0] * (1 - attn_decay * (note_data[...,1] - 1) )

		duration = min(np.size(orig_notes, 0), __hard_limit)
		notes = ema(np.tile(orig_notes, 1), n_smoothing)[:duration]

		# smoothing?
		keys = np.r_[:np.size(notes, 1)]

		fig1 = plt.figure(1, figsize = (5, 5))
		
		grid = AxesGrid(fig1, 111, 
			nrows_ncols = (4, 1),
			axes_pad = 0.05,
			label_mode = "1",
			)

		Plot_Bases = grid[3]

		basis_prob = np.zeros((duration, 12)) 	  # likelihoods
		basis_label = np.zeros((duration, 1)) - 1 # labels
		tension = np.zeros(duration)

		def axis_basis(quanta):
			actives = keys[quanta > 0.0001] + 3 # due to midi pitch nonsense - 0 is C
			weights = quanta[quanta > 0.0001]
			return tens_mod.basis_likelihood(np.vstack((actives, weights)).T)

		def label(base, start, end):
			if start == 0:
				start -= 1
			Plot_Bases.broken_barh([(start + 1, end - start - 1)], (base + 0.25 , 0.5), facecolors = 'red', alpha = 0.3, linewidth = 0)
			basis_label[start + 1:end] = base

		def bar(time):
			Plot_Bases.broken_barh([(time, 0.1)], (0, 12), facecolors = 'red', alpha = 0.7, linewidth = 0)

		def block(base, start, end, color):
			if __block:
				Plot_Bases.broken_barh([(start + 1, end - start)], (base + 0.25, 0.5), facecolors = color, alpha = 1.0, linewidth = 0)
			
		# go through the time slices...
		left = 0
		right = 0
		reach = 0
		candidate = None
		confidence = 0 # SHOULD BE USED

		def notes_in_time(start, end):
			return np.sum(note_data[start:end,:,1])

		def label_basis():
			if candidate == None:
				for base in TensionModule.v_basis[b_curr == np.max(b_curr)]:
					label(base, left - 1, right + 1)
			else:
				label(candidate, left - 1, right + 1)

		def get_cand(notes):
			# return (candidate, confidence)
			return (trim(b_curr, b_ambiguous), np.max(b_curr))

		while right < duration and right < __hard_limit:
			# report
			if right % __report_interval == 0:
				print('{0}/{1}...'.format(right, duration))

			# If we didn't have a candidate, try to check for one
			if candidate == None:
				# get current hypothesis from accumulated notes
				confidence_factor = 1 - 1/(notes_in_time(left, right + 1) + 1)
				section = np.sum(notes[left:right+1], 0)

				b_curr = axis_basis(section) * confidence_factor

				basis_prob[right] = b_curr
				tension[right] = tens_mod.selfTension(section)

				(try_cand, try_conf) = get_cand(b_curr)

				# make sure there is only one candidate, and that it is confident enough
				if np.sum(try_cand) == 1 and try_conf > thresh_conf:
					# found a candidate
					debug_print('got', right, try_cand, try_conf)
					block(-0.5, right - 1, right, 'purple')
					candidate = TensionModule.v_basis[try_cand > b_ambiguous][0]
					confidence = try_conf
				else:
					# no candidate / still ambiguous
					if np.sum(try_cand) > 1:
						debug_print('non', right, 'multiple')
					else:
						debug_print('non', right, try_conf, '<', thresh_conf)
					block(-0.5, right - 1, right, 'blue')
			# If there is a candidate, check to see if the next observed slice follows
			else:
				reach = 0
				similarity = -1

				# attempt to bridge gap if dissimilarity is seen
				while reach < max_gap and right + reach < duration and similarity < persistence:
					# check if the following slice fits the hypothesis
					# section = np.vstack((notes[left:right+1], notes[right + reach]))
					# b_next = axis_basis(np.sum(section, 0))
					b_next = axis_basis(notes[right + reach])
					(try_cand, try_conf) = get_cand(b_next)

					# similarity = matches(trim(b_curr, b_ambiguous, 1), b_next)
					similarity = b_next[candidate]
					debug_print('chk', right, right + reach, 'cnd', candidate, similarity)
					reach += 1

				# exited due to similarity - can extend
				if similarity >= persistence or right + reach >= duration:
					block(-0.5, right - 1, right, 'green')

					debug_print('ext', right, similarity)
					# all's right - we can aggregate this slice
					section = np.sum(notes[left:right+1], 0)
					# basis_prob[right] = b_curr = axis_basis(section)
					basis_prob[right] = b_curr = axis_basis(notes[right])
					tension[right] = tens_mod.selfTension(section)
				# a gap was found and was too large
				elif similarity < persistence and reach >= max_gap:
					block(0, right - 1, right, 'yellow')
					debug_print('rev', right, similarity, list(b_next))
					bar(right)
					right -= 1
					label_basis()
					candidate = None
					left = right + 1

			right += 1

			# back-label

		# label when hitting end
		label_basis()

		# basis_prob = np.apply_along_axis(axis_basis, 1, notes)
		# ema(basis_prob, 0.2)

		grid[0].set_title("{0} group {1}".format(roll.filepath, tempo))
		
		Plot_Bases.locator_params(axis='y', nbins = 12)

		min_note = min([tape.min_note for tape in group])
		max_note = max([tape.max_note for tape in group])

		grid[0].plot(np.r_[:duration] + 0.5, 2 * tension / np.max(tension), 'k')
		grid[1].imshow(notes.T[min_note:max_note + 1],
			interpolation = 'none',
			cmap = plt.cm.Oranges,
			origin = 'lower',
			extent=[0, duration, 0, 12],
			aspect = 0.5 * duration / 24)
		Plot_Bases.imshow(basis_prob.T,
			interpolation = 'none', 
			cmap = plt.cm.Greys,
			origin = 'lower',
			extent=[0, duration, 0, 12],
			aspect = 0.5 * duration / 24)
		grid[2].imshow(basis_label.T,
			interpolation = 'none', 
			cmap = plt.cm.jet,
			origin = 'lower',
			extent=[0, duration, 0, 12],
			aspect = 0.5 * duration / (24 * 3))

		# print(basis_label)

		plt.show()

		# TODO label afterwards, and re-pickle

	# This is really a classification problem that ought to be addressed with the proper tools



	# get it to mark what actions it took - exploring, backlabelling
	# prevent "consecutive self-labelling" for example
	# and also smooth away hiccups

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
if __name__ == '__main__':
	# with PyCallGraph(output=GraphvizOutput(output_file = "BASIS.png")):
	
	# do_basis_label('./mid/bach/aof/can1.mrl', dissonance_metric)
	do_basis_label('./mid/moldau_single.mrl', TensionModule.metric.western)
	do_basis_label('./mid/moldau_accomp.mrl', TensionModule.metric.western)
	# do_basis_label('./mid/bach/aof/can1.mrl', TensionModule.metric.western)
	# do_basis_label('./mid/oldyuanxian2.mrl', TensionModule.metric.western)
	# do_basis_label('./mid/mary.mrl')
	# do_basis_label('./mid/ambigious_test.mrl', TensionModule.metric.western) # lowsim 0.3 - 0.35
	# do_basis_label('./mid/ivivi.mrl', TensionModule.metric.western) # lowsim 0.5



"""
TODO: Confidence in labelling - should be used
TODO: Make sure no OTHER candidate surpasses the first before extending
"""


"""
Entirely hopeless; needs restart.

ON NEXT TRY:
	1: Interleave note activation times in MusicRoll instead of producing the actual timeseries.
	(Timeseries for use with neural model)
	2: Bias towards "sensible" shifts from previous basis (circle of fifths distance)
"""