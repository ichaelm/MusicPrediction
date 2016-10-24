# midinormalizer.py
# Zicheng (Brian) Gao
# Parse MIDI. Different tempos necessitate different sections.

# TODO: Track handling.
# TODO: Dealing with zero-length notes / avoiding percussion

import struct
import pickle
import numpy as np
import os
import mido
from mido import MidiFile, MetaMessage

class MusicRoll:

	def __init__(self, filepath='', labels=[],tapes=[]):
		self.filepath = filepath # filepath of midi file
		self.labels = labels # labels for tapefiles
		self.tapes = tapes # tapefiles - delete before pickling!
		self.tapeamt = len(tapes) # amount of tapefiles
		self.midiFile = MidiFile(filepath) # midi file itself

	def normalize_and_chop(self):
		MidiNormalizer(self, self.midiFile).read()
		del self.midiFile, self.tapes
		pickle.dump(self, open(self.filepath[:-4] + '_info.p', 'bw') )

	# append a new, absolute pitch-only tape
	def appendAbsoluteTape(self, time, tempo):
		tapeName = self.filepath[:-4] + '_data_' + str(self.tapeamt) + '.mrl'
		tape = MusicTape(tapeName, tempo=tempo, hasBasis = False) # make new tape
		self.tapes.append(tape) # tack on this new tape
		self.labels.append( (tapeName, self.tapeamt, False) )
		self.tapeamt += 1
		return tape

# contains on/off information
# The only scenario that would include the creation of a tape would be in initialization or basis identification
class MusicTape:
	NOTE_ON = 1
	NOTE_OFF = 0
	BASIS_CHANGE = 2
	TIME_CHANGE = 3

	# OTHER VARIABLES DETERMINED BY PROCESSING:
	# initialBasis
	# instrument
	def __init__(self, filepath, data=[], tempo=500000, hasBasis=False, startTime=0):
		self.filepath = filepath
		self.data = data
		self.tempo = tempo
		self.hasBasis = hasBasis
		self.startTime = startTime

	def addNoteEvent(self, etype, note, velocity):
		self.data.append( [etype, note, velocity] )

	def addTimeEvent(self, length):
		self.data.append( [MusicTape.TIME_CHANGE, length // 256, length % 256] )

	def finalize(self):
		self.bytecount = len(self.data) * 3
		self.packtype = '>' + str(self.bytecount) + 'B'
		buf = []
		self.data = np.array(self.data).flatten()
		self.data = struct.pack(self.packtype, *self.data)
		print(self.data)
		# self.data = np.array(self.data)
		# self.data = self.data


class MidiNormalizer:
	def __init__(self, roll, midiFile):
		self.roll = roll
		self.midiFile = midiFile
		self.time = 0
		self.tempo = 500000
		self.started = False
		self.activeNotes = {}
		self.noteLengths = []

	def cleanupTimes(self):
		toDelete = []
		# first note on after a note off determines resting time
		for key, note in self.activeNotes.items():
			if note[0] is False:
				note[3] = self.time - note[1] # length from on to off + length from off to next note on
				self.noteLengths.append( ( key, note[1], note[2], note[4], note[3], note[3] - note[2]) )
										#  key  start    notetime velocity fulltime      downtime
				toDelete.append(key)
		for note in toDelete:
			del self.activeNotes[note]

	def noteOn(self, event):
		if event.time is not 0:
			self.time += self.getTicks(event.time)
		# initial activation
		self.activeNotes[event.note] = [True, self.time, 0, 0, event.velocity] # flag, activation tick, on-off duration tick, on-off-next_on tick, velocity
		# boolean is a 'downtime flag': true if note was just turned on, false if off but next note not fired (downtime)
		self.cleanupTimes() # clear downtimes for ready notes
		# print(event.note, event.velocity, '\ton  @ time\t', self.time, 'after', self.getTicks(event.time), 'ticks')

	def noteOff(self, event):
		if event.time is not 0:
			self.time += self.getTicks(event.time)
		if event.note in self.activeNotes:
			self.activeNotes[event.note][0] = False
			self.activeNotes[event.note][2] = self.time - self.activeNotes[event.note][1] # length from on to off
		# print(event.note, '  \toff @ time\t', self.time, 'after', self.getTicks(event.time), 'ticks')

	def getTicks(self, time):
		sec_per_beat = self.tempo / 1000000.0
		sec_per_tick = sec_per_beat / float(self.midiFile.ticks_per_beat)
		ticks = time / sec_per_tick
		return int(ticks)

	def read(self):
		for event in self.midiFile:
			if event.type == 'note_on':
				if event.velocity > 0:
					self.started = True
					self.noteOn(event)
				else:
					self.noteOff(event)
			if event.type == 'note_off':
				self.noteOff(event)
				
			elif isinstance(event, MetaMessage):
				# print(event, event.type)
				if event.type == 'set_tempo':
					self.tempo = event.tempo
					if self.started: # tempo change after file has already begun
						self.normalize_to_tape()
				elif event.type == 'end_of_track':
					self.cleanupTimes() # stragglers / last notes
					self.normalize_to_tape()

	def normalize_to_tape(self):
		### NORMALIZE
		lens = np.array(self.noteLengths).astype(float)
		# normalize note times and on-times by dividing over (minimum of full / round (full / note) )
		ratio = np.round(lens[:,4] / lens[:,2]) # TODO: Catch division by zero for zero-length 'notes'
			# maybe this will be solved when we avoid the percussive, atonal instruments
		factor = np.min(lens[:,4][lens[:,4]!=0]) / ratio
		lens[:,1] = np.round(lens[:,1] / factor)
		lens[:,2] = np.round(lens[:,2] / factor)

		# normalize, convert to tuples of (NOTE, ON_TIME, DURATION, VELOCITY)
		lens = lens.astype(int)[:,0:4]

		### CHANGE TO EVENTS
		# Tuples of TIME, ON/OFF, NOTE, VELOCITY
		events = np.tile(lens[:,0], (4, 2)).T # NOTE by default

		# TIME
		events[:len(lens), 0] = lens[:,1] # note on events
		events[len(lens):, 0] = lens[:,1] + lens[:,2]
		# ON/OFF
		events[:len(lens), 1] = MusicTape.NOTE_ON
		events[len(lens):, 1] = MusicTape.NOTE_OFF
		# VELOCITIES
		events[:len(lens), 3] = lens[:,3] # note on velocities
		events[len(lens):, 3] = 0 # note off velocities

		events = events[events[:,0].argsort()] # Sort tuples by ON_TIME	
		# print(events)
		
		### SEPARATE TIME AND NOTE EVENTS
		# now we iterate through this to produce time events and note events
		tick = 0
		nothings = 0 # count empty time quanta
		tape = self.roll.appendAbsoluteTape(self.time, self.tempo)

		for i in range(0, events.shape[0]):
			if events[i][0] > tick:
				# print(events[i][0] - tick, 'time')
				tape.addTimeEvent(events[i][0] - tick)
				tick = events[i][0]
			# print(events[i], 'note')
			tape.addNoteEvent(events[i][1], events[i][2], events[i][3])
		
		tape.finalize()

		pickle.dump(tape, open(tape.filepath, 'wb'))

def iter_midis_in_path(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mid") or file.endswith(".MID"):
                 yield (os.path.join(root, file), file)

if __name__ == "__main__":
	for path, file in iter_midis_in_path('.'):
		MusicRoll(path).normalize_and_chop()
