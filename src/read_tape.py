# read test for mrl and info
# Zicheng (Brian) Gao

import numpy as np
import pickle
import pprint
import struct
from midinormalizer import *

pp = pprint.PrettyPrinter(indent=4)

# roll = pickle.load(open('mary.mrl', 'rb'))
roll = pickle.load(open('./mid/oldyuanxian2.mrl', 'rb'))
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
	
	pp.pprint(vars(tape))

	# data = struct.unpack(tape.packtype, tape.data)
	# data = np.array(data).reshape(len(data)//3, 3)

	# for event in data:
	# 	if event[0] == MusicTape.NOTE_ON:
	# 		print('Note on:', event[1], 'with velocity', event[2])
	# 	elif event[0] == MusicTape.NOTE_OFF:
	# 		print('Note off:', event[1], 'with velocity', event[2])
	# 	elif event[0] == MusicTape.BASIS_CHANGE:
	# 		print('Basis changed to', event[1])
	# 	elif event[0] == MusicTape.TIME_CHANGE:
	# 		print('Process loops:', 256 * event[1] + event[2])
	# 	else:
	# 		print('Unknown event!')
