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
	def appendAbsoluteTape(self, time, tempo, instrument, channel, unit, min_common):
		tape = MusicTape(data = [], tempo = tempo, hasBasis = False, startTime = time, instrument = instrument) # make new tape
		tape.unit = unit
		tape.min_common = min_common
		self.tapes.append(tape) # tack on this new tape
		self.labels.append(TapeLabel(self.tapeamt, time, tempo, False, channel, unit, min_common))
		self.tapeamt += 1
		return tape

	# dump all tapes, delete tapes, dump roll
	def dump(self, self_contained = False):
		self.self_contained = self_contained
		if not self_contained:
			for label in self.labels:
				label.filename = self.tapes[label.index].filename = "{0}_{1}.mtp".format(self.midipath[:-4], label.index)
				pickle.dump(self.tapes[label.index], open(self.tapes[label.index].filename, 'bw'))

			del self.tapes
		pickle.dump(self, open(self.filepath, 'bw'))

class TapeLabel:
	def __init__(self, index, time, tempo, hasBasis, channel, unit, min_common):
		self.index = index
		self.time = time
		self.tempo = tempo
		self.hasBasis = hasBasis
		self.channel = channel
		self.unit = unit
		self.min_common = min_common

class MusicTape:
	NOTE_ON = 1
	NOTE_OFF = 0
	BASIS_CHANGE = 2
	TIME_CHANGE = 3

	# OTHER VARIABLES DETERMINED BY PROCESSING:
	# initialBasis
	def __init__(self, data=[], tempo=500000, hasBasis=False, startTime=0, instrument=0):
		self.data = data
		self.ticks = 0 # how many ticks
		self.tempo = tempo
		self.hasBasis = hasBasis
		self.startTime = startTime
		self.instrument = instrument
		self.min_note = 127
		self.max_note = 0

	def addNoteEvent(self, etype, note, velocity):
		self.data.append( [etype, note, velocity] )
		if note > self.max_note:
			self.max_note = note
		if note < self.min_note:
			self.min_note = note

	def addTimeEvent(self, length):
		self.data.append( [MusicTape.TIME_CHANGE, length // 256, length % 256] )
		self.ticks += length

	def finalize(self):
		self.bytecount = len(self.data) * 3
		self.packtype = '>' + str(self.bytecount) + 'B'
		self.data = np.array(self.data).flatten()
		self.data = struct.pack(self.packtype, *self.data)

	def unpack_data(self):
		data = struct.unpack(self.packtype, self.data)
		return np.array(data).reshape(len(data)//3, 3)

	def timeseries(self, relative = False):
		# time, note, [velocity, newness]
		if relative:
			timeseries = np.zeros((self.ticks + 1, self.max_note - self.min_note + 1, 2))
		else:
			timeseries = np.zeros((self.ticks + 1, 127, 2))

		def relative_note(absolute_note):
			if relative:
				return absolute_note - self.min_note
			else:
				return absolute_note

		i = 0
		data = self.unpack_data()
		for event in data:
			if event[0] == MusicTape.NOTE_ON:
				# note event
				timeseries[i, relative_note(event[1]), 0] = event[2] / 127.0
				timeseries[i, relative_note(event[1]), 1] = 1
				# print('Note on:', relative_note(event[1]), 'with velocity', event[2])
			elif event[0] == MusicTape.NOTE_OFF:
				timeseries[i, relative_note(event[1]), 0] = event[2] / 127.0
				# print('Note off:', relative_note(event[1]), 'with velocity', event[2])
			elif event[0] == MusicTape.BASIS_CHANGE:
				# print('Basis changed to', event[1])
				pass # TODO - not implemented
			elif event[0] == MusicTape.TIME_CHANGE:
				next_pointer = i + 256 * event[1] + event[2]
				# only carry over note slice
				timeseries[i + 1:next_pointer + 1, :, 0] = timeseries[i, :, 0]
				i = next_pointer
				# print('Process loops:', 256 * event[1] + event[2])
			else:
				print('Unknown event!')

		return timeseries