import pickle
import numpy as np
import struct

class MusicRoll:
	def __init__(self, midipath='', labels=[],tapes=[]):
		self.midipath = midipath # midipath of midi file
		self.filepath = self.midipath[:-4] + '.mrl'
		self.labels = labels # labels for tapefiles
		self.tapes = tapes # tapefiles - delete before pickling!
		self.tapeamt = len(tapes) # amount of tapefiles

	# append a new, absolute pitch-only tape
	def appendAbsoluteTape(self, time, tempo, instrument, channel, unit):
		tape = MusicTape(data = [], tempo = tempo, hasBasis = False, startTime = time, instrument = instrument) # make new tape
		self.tapes.append(tape) # tack on this new tape
		self.labels.append(TapeLabel(self.tapeamt, time, tempo, False, channel, unit))
		self.tapeamt += 1
		return tape

	# dump all tapes, delete tapes, dump roll
	def dump(self):
		for label in self.labels:
			label.filename = self.tapes[label.index].filename = "{0}_{1}.mtp".format(self.midipath[:-4], label.index)
			pickle.dump(self.tapes[label.index], open(self.tapes[label.index].filename, 'bw'))

		del self.tapes
		pickle.dump(self, open(self.filepath, 'bw'))

class TapeLabel:
	def __init__(self, index, time, tempo, hasBasis, channel, unit):
		self.index = index
		self.time = time
		self.tempo = tempo
		self.hasBasis = hasBasis
		self.channel = channel
		self.unit = unit

class MusicTape:
	NOTE_ON = 1
	NOTE_OFF = 0
	BASIS_CHANGE = 2
	TIME_CHANGE = 3

	# OTHER VARIABLES DETERMINED BY PROCESSING:
	# initialBasis
	# instrument
	def __init__(self, data=[], tempo=500000, hasBasis=False, startTime=0, instrument=0):
		self.data = data
		self.tempo = tempo
		self.hasBasis = hasBasis
		self.startTime = startTime
		self.instrument = instrument

	def addNoteEvent(self, etype, note, velocity):
		self.data.append( [etype, note, velocity] )

	def addTimeEvent(self, length):
		self.data.append( [MusicTape.TIME_CHANGE, length // 256, length % 256] )
		# self.data.append( [MusicTape.TIME_CHANGE, length] )

	def finalize(self):
		self.bytecount = len(self.data) * 3
		self.packtype = '>' + str(self.bytecount) + 'H'
		buf = []
		# print(self.data)
		self.data = np.array(self.data).flatten()
		self.data = struct.pack(self.packtype, *self.data)
		pass
