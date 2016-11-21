# midinormalizer.py
# Zicheng (Brian) Gao
# Parse MIDI. Different tempos necessitate different sections.

import numpy as np
from mido import MidiFile, MetaMessage
from MusicRoll import *

verbosity = 0

print_event = False
show_note_events = False
show_note_info = False
show_raw_time_events = False
show_onset_time_diffs = False
show_full_length_occurrences = False
show_normalized_events = False
show_tape_data = False

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
	def channel_done(self, n, instrument = 0):
		self.channels[n].cleanupTimes()
		if not self.channels[n].convert_to_events():
			del self.channels[n]
			return False
		# TODO: Avoid note(s) from being divided by this function
		# in the case that there is a tempo change in the middle of notes
		
		if self.tempo not in self.final_channels:
			self.final_channels[self.tempo] = []
		self.final_channels[self.tempo].append(self.channels[n])

		# replace with new
		self.channels[n] = ChannelNormalizer(self, n, self.time, instrument = instrument)
		return True

	def extract_events(self):
		for event in self.midiFile:
			if print_event:
				print(event)
			# sec per beat = tempo / million
			# sec per tick = sec per beat / ticks per beat
			# ticks = time / sec per tick = time / (spb/tpb) = tpb * t / spb = tpb * t / (tmp / mill) = tpb * t * mill / tmp
			self.time += round(event.time * float(self.midiFile.ticks_per_beat) * 1000000.0 / self.tempo)

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
				# channel 9 is always for percussion (in GM standard MIDI)
				if event.type == 'sysex':
					print("Unhandled sysex:",event)
				elif event.channel != 9:
					if event.channel not in self.channels:
						self.channels[event.channel] = ChannelNormalizer(self, event.channel, self.time)
					self.channels[event.channel].handle_event(event)

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

			if show_full_length_occurrences:
				print("Full Length Occurrences:")
				print(self.fullen_dist[tempo])

			total_notes = np.sum(self.fullen_dist[tempo][:,1])
			print("Total notes:", total_notes)
			
			# find smallest of most common lengths
			common = self.fullen_dist[tempo][self.fullen_dist[tempo][:,1].argsort(-1)][::-1]
			common = common[:median_index(common[:,1], total_notes)]
			mincommon = np.min(common[:,0])

			# try note lengths smaller than median of occurrences
			# COST = (LEN - closestmultiple(LEN, CANDIDATE)) * COUNT
			# TODO: Penalize excessively small intervals, such as "1" ?
			i = median_index(self.fullen_dist[tempo][:,1], total_notes)

			# second col will be replaced with loss
			# third col will correspond to possible next candidate
			candidates = np.c_[self.fullen_dist[tempo][:i], np.zeros(i).T.astype(int)]

			# initial pass
			# ensure there are no zero-length candidates, first
			candidates = candidates[candidates[...,0] != 0]
			next_r = self.best_candidate_mut(candidates, self.fullen_dist[tempo])
			# cannibalize the candidates, begin refinement cycle
			candidates = np.vstack((next_r[0], [next_r[0,2], 0, 0]))

			loss = np.sum(self.fullen_dist[tempo][:,0]) # maximal
			threshold = chop_loss_percent * loss # adjust? 0.5 percent loss permissible
			converged = False

			# refinement passes - TODO (fix hackneyed program flow)
			while candidates[0,1] > threshold and np.all(candidates[...,0] > 0) and not converged:
				# get loss values and potential next candidates
				next_r = self.best_candidate_mut(candidates, self.fullen_dist[tempo])
				converged = candidates[1,1] >= candidates[0,1] 			 # did the loss actually get worse?
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
	def best_candidate_mut(self, candidates, full_lengths):
		# return best candidate, writing in loss into input array of candidates
		for candidate_row in candidates:
			if candidate_row[0] == 0:
				print("Zero candidate!!", candidates)

			# closest multiple, error weighted by note length occurence frequency
			adjusted = np.round(full_lengths[:,0] / candidate_row[0])
			
			# adjusted values do not go to zero - not here, anyway
			adjusted[adjusted == 0] += 1

			diffs = (full_lengths[:,0] - candidate_row[0] * adjusted).astype(int)
			candidate_row[2] = np.max(np.abs(diffs))

			# loss
			candidate_row[1] = np.sum(np.abs(diffs) * full_lengths[:,1])
		return candidates[np.min(candidates[:,1])==candidates[:,1]]

def median_index(array, maxm):
	i = 0
	count = 0
	while count < (maxm / 2):
		count += array[i]
		i += 1
	return i

class ChannelNormalizer:
	def __init__(self, owner, num, start_time, instrument = 0):
		self.num = num
		self.started = False
		self.owner = owner
		self.instrument = instrument # piano - default
		self.activeNotes = {}
		self.noteLengths = []
		self.events = []
		self.start_time = start_time
		self.volume = [1.000, 1.000] # one for volume, one for expression

	def cleanupTimes(self, delete = True):
		toDelete = []
		for key, note in self.activeNotes.items():
			if note[0] is False:
				note[3] = self.owner.time - note[1]
										#  key  start    notetime velocity fulltime      downtime
				self.noteLengths.append( ( key, note[1], note[2], note[4], note[3], note[3] - note[2]) )
				toDelete.append(key)
		if delete:
			for note in toDelete:
				del self.activeNotes[note]

	def noteOn(self, event):
		if show_note_events:
			print(event)
		# initial activation
		# flag, activation tick, on-off duration tick, on-off-next_on tick, discretized velocity
		self.cleanupTimes() # clear downtimes for ready notes
		self.activeNotes[event.note] = [True, self.owner.time, 0, 0, round(event.velocity * self.volume[0] * self.volume[1])]
		# self.cleanupTimes() # clear downtimes for ready notes
		# boolean is a 'downtime flag': true if note was just turned on, false if off but next note not fired (downtime)

	def noteOff(self, event):
		if show_note_events:
			print(event)
		if event.note in self.activeNotes:
			self.activeNotes[event.note][0] = False
			self.activeNotes[event.note][2] = self.owner.time - self.activeNotes[event.note][1] # length from on to off

	def handle_event(self, event):
		# handle an event
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
			if event.program != self.instrument:
				if self.started:
					# self.owner.final_channels[self.owner.tempo].append(self)
					print("Channel {0} changed instrument to {1}".format(self.num, event.program))
					self.owner.channel_done(self.num, instrument = event.program)
				else:
					self.instrument = event.program
					print("Channel ", event.channel, "Instrument changed to ", event.program)
		elif event.type == 'pitchwheel':
			pass # TODO
			# in honesty, this should probably be ignored
			# as we want to consider "music" and not "effects"
		elif event.type == 'control_change':
			if event.control == 7: # track volume
				self.volume[0] = event.value / 127.0
			elif event.control == 11: # track expression - percentage of volume
				self.volume[1] = event.value / 127.0
			elif event.control == 120 or event.control == 123:
				pass # TODO all notes off
			# all controls reset - also should affect pitchwheel
		else:
			print("Warning: Type not covered:", event)
			
	def convert_to_events(self):
		# Convert note history to event list
		print("## ## ## Channel", self.num)

		lens = np.array(self.noteLengths).astype(float)

		if np.size(lens) is 0:
			print("Empty channel!")
			return False
		
		if show_note_info:
			print("lengths:\n\t[NOTE ON_TIME NOTE_DURATION VELOCITY FULLTIME DOWNTIME]")
			for row in lens.astype(int):
				print('\t{}'.format(row))

		if show_onset_time_diffs:
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

		# make sure on is before off, then sort by on-time
		events = events[np.lexsort((events[:,1], events[:,0]))]
		if show_raw_time_events:
			print("Raw-time events:")
			print("\t[TIME ON/OFF NOTE VELOCITY]")
			for event in events:
				print('\t', event)

		self.events = events
		return True

	def normalize_to_tape(self, unit_len, min_common):
		# normalize event times
		events = self.events
		events[:, 0] = np.round(events[:, 0] / unit_len)
		self.start_time = np.min(events[:, 0])# time of first event onset
		
		# Separate time and note events for easy processing
		tape = self.owner.roll.appendAbsoluteTape(
			self.start_time,
			self.owner.tempo,
			self.instrument,
			self.num,
			unit_len,
			min_common) # note count

		print("Event count:", np.size(events, 0))

		if show_normalized_events:
			print("Events:")
			for event in events:
				print(event)

		# begins at first note. start_time informs this
		tick = self.start_time
		for i in range(0, events.shape[0]):
			if events[i][0] > tick:
				# tape.addTimeEvent(events[i][0] - tick, tick)
				# NO TIME EVENTS - too much of a hassle
				tick = events[i][0]
			tape.addNoteEvent(events[i][1], events[i][2], events[i][3], tick)

		if show_tape_data:
			print("Tape data:")
			for event in tape.data:
				print(event)
		
		print("Finished channel", self.num)
		tape.finalize()

"""
Possible issues:
	after a tempo change, the start-time tick count is no longer accurate
"""

import hashlib

def md5():
    hash_md5 = hashlib.md5()
    with open('midinormalizer.py', "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return str(hash_md5.hexdigest())

if __name__ == '__main__':
	print(md5())