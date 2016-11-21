from MusicRoll import *

roll = pickle.load(open('./mid/channel_sep.mrl', 'rb'))

for tempo, group in roll.get_tape_groups().items():
	notes = roll.combine_notes(group)[...,0]
	keys = np.r_[:np.size(notes, 1)]
	for row in notes:
		print(keys[row != 0])
	