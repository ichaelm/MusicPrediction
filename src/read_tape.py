# read test for mrl and mtp
# Zicheng (Brian) Gao

import numpy as np
import pickle
import pprint
from midinormalizer import *

pp = pprint.PrettyPrinter(indent=4)

# roll = pickle.load(open('./mid/mary.mrl', 'rb'))
# roll = pickle.load(open('./mid/oldyuanxian2.mrl', 'rb'))
roll = pickle.load(open('./mid/two_channel_test.mrl', 'rb'))
pp.pprint(vars(roll))

time = 0
# all tapes belonging to the same roll should have the same unit lengths
for label in roll.labels:
	pp.pprint(vars(label))

	if roll.self_contained:
		# use label to find file to find tape
		tape = roll.tapes[label.index]
	else:
		# use label to find tape
		tape = pickle.load( open(label.filename, 'rb') )
	
	# pp.pprint(vars(tape))

	print(tape.ticks, "ticks")

	timeseries = tape.timeseries(relative = True)

	print(np.size(timeseries, 0))
	
	for row in timeseries[:, :, 0]:
		print(row)
	