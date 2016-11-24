from MusicRoll import *
import pprint

filename = './mid/bach/aof/can1.mrl'
# filename = './mid/two_channel_test.mrl'

pp = pprint.PrettyPrinter(indent=4)
roll = pickle.load(open(filename, 'rb'))
pp.pprint(vars(roll))

for tempo, group in roll.get_tape_groups().items():
	sum_notes = 0

	for tape in group:
		print(tape, tape.notes, "notes")
		series = tape.timeseries()
		sum_notes += np.sum(series[...,1])
		print(np.sum(series[122:144,:,1]))

	notes_data = MusicRoll.combine_notes(group)

	print(np.sum(notes_data[...,1]))
	print(np.sum(notes_data[122:144,:,1]))
	print(sum_notes)