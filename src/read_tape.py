# read test for mrl and mtp
# Zicheng (Brian) Gao

import numpy as np
import pickle
import pprint
from midinormalizer import *

pp = pprint.PrettyPrinter(indent=4)

roll = pickle.load(open('./mid/mary.mrl', 'rb'))
# roll = pickle.load(open('./mid/oldyuanxian2.mrl', 'rb'))
pp.pprint(vars(roll))

time = 0
for label in roll.labels:
	pp.pprint(vars(label))

	if roll.self_contained:
		# use label to find file to find tape
		tape = roll.tapes[label.index]
	else:
		# use label to find tape
		tape = pickle.load( open(label.filename, 'rb') )
	
	# pp.pprint(vars(tape))
	data = tape.unpack_data()

	print(tape.ticks, "ticks")

	# time, note, [velocity, newness]
	# timeseries = np.zeros((tape.ticks + 1, tape.max_note - tape.min_note + 1, 2))
	timeseries = np.zeros((tape.ticks + 1, 127, 2))
	pointer = 0
	# go down the rows of the timeseries

	def relative_note(absolute_note):
		return absolute_note
		# return absolute_note - tape.min_note

	for event in data:
		if event[0] == MusicTape.NOTE_ON:
			# note event
			timeseries[pointer, relative_note(event[1]), 0] = event[2] / 127.0
			timeseries[pointer, relative_note(event[1]), 1] = 1
			# print('Note on:', relative_note(event[1]), 'with velocity', event[2])
		elif event[0] == MusicTape.NOTE_OFF:
			timeseries[pointer, relative_note(event[1]), 0] = event[2] / 127.0
			# print('Note off:', relative_note(event[1]), 'with velocity', event[2])
		elif event[0] == MusicTape.BASIS_CHANGE:
			# print('Basis changed to', event[1])
			pass # TODO - not implemented
		elif event[0] == MusicTape.TIME_CHANGE:
			next_pointer = pointer + 256 * event[1] + event[2]
			# only carry over note slice
			timeseries[pointer + 1:next_pointer + 1, :, 0] = timeseries[pointer, :, 0]
			pointer = next_pointer
			# print('Process loops:', 256 * event[1] + event[2])
		else:
			print('Unknown event!')
	
	# for row in timeseries[:, :, :]:
		# print(row)
	