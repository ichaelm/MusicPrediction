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

	def set_hash(self, md5):
		self.md5 = md5

	# append a new, absolute pitch-only tape
	def appendAbsoluteTape(self, time, tempo, instrument, channel, unit, min_common):
		tape = MusicTape(data = [], tempo = tempo, has_basis = False, start_time = time, instrument = instrument) # make new tape
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

	def get_tape_groups(self):
		# iterate events across all tapes that belong to this roll

		# "reconstruct midi" - put all events into tempogroups
		tempogroups = {}
		for label in self.labels:
			# find tape
			if self.self_contained:
				tape = self.tapes[label.index]
			else:
				tape = pickle.load( open(label.filename, 'rb') )
			# add to tempogroups
			if tape.tempo not in tempogroups:
				tempogroups[tape.tempo] = [tape]
			else:
				tempogroups[tape.tempo].append(tape)

		return tempogroups

	@staticmethod
	def combine_notes(tapelist):
		# Interleave note and time events of tapes - WARNING: only returns note events
		assert all([tapelist[0].tempo == tape.tempo for tape in tapelist])

		tapelist = sorted(tapelist, key = lambda tape: tape.start_time)

		end = 0
		for tape in tapelist:
			endq = tape.start_time + tape.length
			if endq > end:
				end = endq

		beginning = tapelist[0].start_time

		# print("Begin ", beginning)
		# print("End   ", end)
		# print("Length", end - beginning)
		output = np.zeros((end - beginning, 127, 2))

		for tape in tapelist:
			src = tape.timeseries()[:tape.length]
			# print("Tapelen", tape.length)
			output[tape.start_time - beginning:tape.start_time - beginning + tape.length] += src

		return output

		# this will run into problems if tapes in the group don't end at the same time
		# right now we just want a working thing, though, and such a situation
		# is not so likely to be common (program changes, for example)

		# for row in output[...,0]:
			# print(row)

class TapeLabel:
	def __init__(self, index, time, tempo, has_basis, channel, unit, min_common):
		self.index = index
		self.time = time
		self.tempo = tempo
		self.has_basis = has_basis
		self.channel = channel
		self.unit = unit
		self.min_common = min_common

class MusicTape:
	NOTE_ON = 1
	NOTE_OFF = 0
	BASIS_CHANGE = 2
	TIME_CHANGE = 3
	BYTES = 5

	# OTHER VARIABLES DETERMINED BY PROCESSING:
	# initialBasis
	def __init__(self, data=[], tempo=500000, has_basis=False, start_time=0, instrument=0):
		self.data = data
		self.ticks = 0 # how many ticks
		self.tempo = tempo
		self.has_basis = has_basis
		self.start_time = start_time
		self.instrument = instrument
		self.min_note = 127
		self.max_note = 0

	def addNoteEvent(self, etype, note, velocity, time):
		self.data.append( [etype, note, velocity, time // 256, time % 256] )
		if note > self.max_note:
			self.max_note = note
		if note < self.min_note:
			self.min_note = note

	def addTimeEvent(self, length, time):
		self.data.append( [MusicTape.TIME_CHANGE, length // 256, length % 256, time] )
		# self.ticks += length

	def finalize(self):
		self.bytecount = len(self.data) * MusicTape.BYTES
		self.packtype = '>' + str(self.bytecount) + 'B'
		self.data = np.array(self.data)

		self.ticks = np.max(self.data[:,3] * 256 + self.data[:,4])
		self.length = self.ticks - self.start_time + 1

		self.data = self.data.flatten()
		self.data = struct.pack(self.packtype, *self.data)

	def unpack_data(self):
		data = struct.unpack(self.packtype, self.data)
		return np.array(data).reshape(len(data)//MusicTape.BYTES, MusicTape.BYTES)

	def timeseries(self, relative = False):
		# extract timeseries of positional note velocities for feeding into neural network
		# time, note, [velocity, newness]
		if relative:
			timeseries = np.zeros((self.length, self.max_note - self.min_note + 1, 2))
		else:
			timeseries = np.zeros((self.length, 127, 2))

		def __get_time(event):
			return event[3] * 256 + event[4]

		def relative_note(absolute_note):
			if relative:
				return absolute_note - self.min_note
			else:
				return absolute_note

		i = 0
		data = self.unpack_data()
		for event in data:
			# move up to time
			if i < __get_time(event) - self.start_time:
				next_pointer = __get_time(event) - self.start_time
				# only carry over note slice
				timeseries[i + 1:next_pointer + 1, :, 0] = timeseries[i, :, 0]
				i = next_pointer

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

			# elif event[0] == MusicTape.TIME_CHANGE:
				# next_pointer = i + 256 * event[1] + event[2]
				# only carry over note slice
				# timeseries[i + 1:next_pointer + 1, :, 0] = timeseries[i, :, 0]
				# i = next_pointer
				# print('Process loops:', 256 * event[1] + event[2])
			else:
				print('Unknown event!')

		return timeseries