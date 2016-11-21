import os
import midinormalizer
from mido import MidiFile, MetaMessage
from MusicRoll import *

def iter_midis_in_path(folder_path):
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			if file.endswith(".mid") or file.endswith(".MID"):
				 yield (os.path.join(root, file), file)

def perform(path):
	print("Processing '{0}'".format(path))
	roll = MusicRoll(path, labels = [], tapes = [])
	midi = MidiFile(path)
	midinormalizer.MidiNormalizer(roll, midi).normalize(chop_loss_percent = 0.002) # 0.2 percent
	roll.set_hash(midinormalizer.md5())
	roll.dump(self_contained = False)

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
if __name__ == "__main__":
	# with PyCallGraph(output=GraphvizOutput()):
	for path, file in iter_midis_in_path('.'):

		roll_name = path[:-4] + '.mrl'
		# no music roll file?
		if not os.path.isfile(roll_name):
			perform(path)
		else:
			# file is outdated?
			old_roll = pickle.load(open(roll_name, 'rb'))
			if not (hasattr(old_roll, 'md5') and old_roll.md5 == midinormalizer.md5()):
				perform(path)
			else:
				print("Skipping '{0}'".format(file))



