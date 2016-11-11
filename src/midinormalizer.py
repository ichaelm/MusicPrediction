# midinormalizer.py
# Zicheng (Brian) Gao
# Parse MIDI. Different tempos necessitate different sections.

# TODO: Dealing with zero-length notes / avoiding percussion

import numpy as np
import os
from mido import MidiFile, MetaMessage
from MusicRoll import *

verbosity = 0
outlier_threshold = 0.001

class MidiNormalizer:
	def __init__(self, roll, midiFile):
		self.roll = roll
		self.midiFile = midiFile
		self.time = 0
		self.tempo = 500000
		self.started = False
		self.channels = {}
		# channel lists grouped by tempo
		self.final_channels = {}
		# also grouped by tempo... a dictionary of dictionaries
		self.fullen_dist = {}

	# finalize channel
	def channel_done(self, n):
		self.channels[n].cleanupTimes()
		self.channels[n].convert_to_events()
		# TODO: Avoid note(s) from being divided by this function
		# in the case that there is a tempo change in the middle of notes
		
		if self.tempo not in self.final_channels:
			self.final_channels[self.tempo] = []
		self.final_channels[self.tempo].append(self.channels[n])

		# replace with new
		self.channels[n] = ChannelNormalizer(self, n, self.time)

	def extract_events(self):
		for event in self.midiFile:
			self.time += self.getTicks(event.time)
			if isinstance(event, MetaMessage):
				if event.type == 'set_tempo':
					self.tempo = event.tempo
					# tempo change after file has already begun
					if self.started:
						print("Performing tempo separation")
						for n, channel in self.channels.items():
							self.channel_done(n)
				elif event.type == 'end_of_track':
					print("Total time:", self.time)
					for n, channel in self.channels.items():
						self.channel_done(n)
			# have we seen this channel before?
			else:
				try:
					# channel 9 is always for percussion (in GM standard MIDI)
					if event.type == 'sysex':
						print("Unhandled sysex:",event)
					elif event.channel != 9:
						if event.channel not in self.channels:
							self.channels[event.channel] = ChannelNormalizer(self, event.channel, self.time)
						self.channels[event.channel].handle_event(event)
				except AttributeError as e:
					print(event, e)

	# tempo is a global phenomenon - this takes care of inserting to dictionary
	def count_full_length(self, length):
		if self.tempo not in self.fullen_dist:
			self.fullen_dist[self.tempo] = {}

		if length in self.fullen_dist[self.tempo]:
			self.fullen_dist[self.tempo][length] += 1
		else:
			self.fullen_dist[self.tempo][length] = 1

	# Finish - perform normalization
	def normalize(self, chop_loss_percent):
		self.extract_events() # TODO - maybe bad practice to forcibly run this function here

		# get rid of empty channel recordings
		for tempo, tempogroup in self.final_channels.items():
			for channel in tempogroup:
				if not channel.started:
					tempogroup.remove(channel)

		# read in all full-length distributions for each tempo grouping
		for tempo, tempogroup in self.final_channels.items():

			self.fullen_dist[tempo] = np.array([
				(length, occur) for length, occur in self.fullen_dist[tempo].items()
				]).astype(int)

			self.fullen_dist[tempo] = self.fullen_dist[tempo][self.fullen_dist[tempo][:,0].argsort()]

			# if verbosity > 0:
				# print("Minimum full length:", unit_len)
			if verbosity > 1:
				print("Full Length Occurences:")
				print(self.fullen_dist[tempo])

			total_notes = np.sum(self.fullen_dist[tempo][:,1])
			print("Total notes:", total_notes)
			
			# most common lengths - sort according to occurrence, chop
			common = self.fullen_dist[tempo][self.fullen_dist[tempo][:,1].argsort(-1)][::-1]
			i = median_index(common[:,1], total_notes)
			# smallest of most common lengths
			mincommon = np.min(common[:,0])

			# find the "median" occurrence of notes and try every length shorter than that
			# mismatch metric penalizes misfits,
			# MISMATCH = (LEN - closestmultiple(LEN, CANDIDATE)) * COUNT
			# TODO also excessively small intervals, such as "1"
			i = median_index(self.fullen_dist[tempo][:,1], total_notes)

			# cutoff index is i
			# second col will be replaced with loss
			# third col will correspond to possible next candidate
			candidates = np.c_[self.fullen_dist[tempo][:i], np.zeros(i).T.astype(int)]

			# initial pass
			self.candidate_loss_mut(candidates, self.fullen_dist[tempo])
			
			# cannibalize the candidates, begin refinement cycle
			next_r = candidates[np.min(candidates[:,1])==candidates[:,1]]
			candidates = np.vstack((next_r[0], [next_r[0,2], 0, 0]))

			loss = np.sum(self.fullen_dist[tempo][:,0]) # maximal
			threshold = chop_loss_percent * loss # adjust? 0.5 percent loss permissible
			converged = False

			# refinement passes - TODO (fix hackneyed program flow)
			while candidates[0,1] > threshold and np.all(candidates[:,0] > 0) and not converged:
				# get loss values and potential next candidates
				self.candidate_loss_mut(candidates, self.fullen_dist[tempo])
				next_r = candidates[np.min(candidates[:,1])==candidates[:,1]] # get best candidate
				converged = candidates[1,1] >= candidates[0,1] 				# did the loss actually get worse?
				candidates = np.vstack((next_r[0], [next_r[0,2], 0, 0])) # set up check for potential next candidate

			# candidates have been obtained
			# unit length should be halved to distinguish sequential notes from sustained notes
			# (allow note decay)
			unit_len = candidates[0,0] // 2
			# express minimum common length in terms of unit_length
			mincommon //= unit_len

			print("Unit length for tempo group {0} is {1},\n\t with {2} times unit_len being most common".format(tempo, unit_len, mincommon))

			# adjust all channels in the tempo group
			for channel in tempogroup:
				channel.normalize_to_tape(unit_len, mincommon)

	# also a mutating operation
	def candidate_loss_mut(self, candidates, full_lengths):
		for candidate_row in candidates:
			# closest multiple, error weighted by note length occurence frequency
			loss = full_lengths[:,0]
			loss = (loss - candidate_row[0] * np.round(loss / candidate_row[0])).astype(int)
			candidate_row[2] = np.max(np.abs(loss))
			loss = np.sum(np.abs(loss) * full_lengths[:,1])
			candidate_row[1] = loss
		return loss

	def getTicks(self, time):
		sec_per_beat = self.tempo / 1000000.0
		sec_per_tick = sec_per_beat / float(self.midiFile.ticks_per_beat)
		ticks = time / sec_per_tick
		return round(ticks)

def median_index(array, maxm):
	i = 0
	count = 0
	while count < (maxm / 2):
		count += array[i]
		i += 1
	return i

def iter_midis_in_path(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mid") or file.endswith(".MID"):
                 yield (os.path.join(root, file), file)

class ChannelNormalizer:
	def __init__(self, owner, num, start_time):
		self.num = num
		self.started = False
		self.owner = owner
		self.instrument = 0 # piano - default
		self.activeNotes = {}
		self.noteLengths = []
		self.events = []
		self.start_time = start_time
		self.volume = [1.000, 1.000] # one for volume, one for expression

	def cleanupTimes(self):
		toDelete = []
		for key, note in self.activeNotes.items():
			if note[0] is False:
				note[3] = self.owner.time - note[1]
										#  key  start    notetime velocity fulltime      downtime
				self.noteLengths.append( ( key, note[1], note[2], note[4], note[3], note[3] - note[2]) )
				toDelete.append(key)
		for note in toDelete:
			del self.activeNotes[note]

	def noteOn(self, event):
		# initial activation
		# flag, activation tick, on-off duration tick, on-off-next_on tick, discretized velocity
		self.activeNotes[event.note] = [True, self.owner.time, 0, 0, round(event.velocity * self.volume[0] * self.volume[1])]
		# boolean is a 'downtime flag': true if note was just turned on, false if off but next note not fired (downtime)
		self.cleanupTimes() # clear downtimes for ready notes

	def noteOff(self, event):
		if event.note in self.activeNotes:
			self.activeNotes[event.note][0] = False
			self.activeNotes[event.note][2] = self.owner.time - self.activeNotes[event.note][1] # length from on to off

	def handle_event(self, event):
		if event.type == 'note_on':
			if event.velocity > 0:
				self.started = True
				self.noteOn(event)
			else:
				self.noteOff(event)
		elif event.type == 'note_off':
			self.noteOff(event)
		# TODO may be an instrument change
		elif event.type == 'program_change':
			if self.started:
				owner.final_channels.append(self)
				owner.channel_done(self.num)
			else:
				self.instrument = event.program
				print("Channel ", event.channel, "Instrument changed to ", event.program)
		elif event.type == 'pitchwheel':
			pass # TODO
		elif event.type == 'control_change':
			if event.control == 7: # track volume
				self.volume[0] = event.value / 127.0
			elif event.control == 11: # track expression - percentage of volume
				self.volume[1] = event.value / 127.0
		else:
			print("Warning: Type not covered:", event)
			
	def convert_to_events(self):
		global verbosity
		
		print("## ## ## Channel", self.num)
		
		lens = np.array(self.noteLengths).astype(float)
		
		if verbosity > 5:
			print("lengths:\n\t[NOTE ON_TIME NOTE_DURATION VELOCITY FULLTIME DOWNTIME]")
			for row in lens.astype(int):
				print('\t{}'.format(row))

		if verbosity > 2:
			print("Time diffs between onsets:")
			print(np.diff(lens.astype(int)[:,1]))

		# normalize note times and on-times by dividing over (minimum of full / round (full / note) )
		for full_length in lens[:-1,4]:
			self.owner.count_full_length(full_length)
		
		# normalize, convert to tuples of (NOTE, ON_TIME, DURATION, VELOCITY)
		lens = lens.astype(int)[:,0:4]

		# Change values to event values
		events = np.tile(lens[:,0], (4, 2)).T # NOTE by default
		events[:len(lens), 0] = lens[:,1] # note on time
		events[len(lens):, 0] = lens[:,1] + lens[:,2]
		events[:len(lens), 1] = MusicTape.NOTE_ON
		events[len(lens):, 1] = MusicTape.NOTE_OFF
		events[:len(lens), 3] = lens[:,3] # note on velocities
		events[len(lens):, 3] = 0 # note off velocities

		# Sort by on-time
		events = events[events[:,0].argsort()]
		if verbosity > 2:
			print("Raw-time events:")
			print("\t[TIME ON/OFF NOTE VELOCITY]")
			for event in events:
				print('\t', event)

		self.events = events

	def normalize_to_tape(self, unit_len, min_common):
		# normalize times
		events = self.events
		events[:, 0] = np.round(events[:, 0] / unit_len)
		
		# Separate time and note events for easy processing
		tape = self.owner.roll.appendAbsoluteTape(
			self.start_time,
			self.owner.tempo,
			self.instrument,
			self.num,
			unit_len,
			min_common) # note count

		print("Event count:", np.size(events, 0))

		if verbosity > 10:
			print("Events:")
			for event in events:
				print(event)

		tick = 0
		for i in range(0, events.shape[0]):
			if events[i][0] > tick:
				tape.addTimeEvent(events[i][0] - tick)
				tick = events[i][0]
			tape.addNoteEvent(events[i][1], events[i][2], events[i][3])
		
		print("Finished channel", self.num)
		tape.finalize()

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
if __name__ == "__main__":
	# with PyCallGraph(output=GraphvizOutput()):
	for path, file in iter_midis_in_path('.'):
		roll = MusicRoll(path, labels = [], tapes = [])
		midi = MidiFile(path)

		MidiNormalizer(roll, midi).normalize(chop_loss_percent = 0.002) # 0.2 percent
		roll.dump(self_contained = False)
