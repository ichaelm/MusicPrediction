import midi
import os
import pickle
import array

class RCFF(object):
    
    instrument = -1
    pitches = []
    midi_file_path = ''

    def __init__(self, pitches, instrument=-1, midi_file_path=''):
        self.pitches = pitches
        self.instrument = instrument
        self.midi_file_path = midi_file_path

def midi_files_iter(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mid") or file.endswith(".MID"):
                 yield (os.path.join(root, file))

def extract_notes(track):
    notes = [] # [(time, length, pitch)]
    time = 0
    pitch_started = {}
    instrument = -1
    found_instrument = False
    for event in track:
        #print(event)
        time += event.tick
        if not found_instrument and (type(event) is midi.ProgramChangeEvent):
            found_instrument = True
            if (event.data[0] >= 96):
                raise RuntimeError('atonal')
            instrument = event.data[0]
        if (type(event) is midi.NoteOnEvent):
            pitch_started[event.pitch] = time
        if (type(event) is midi.NoteOffEvent):
            try:
                start_time = pitch_started[event.pitch]
                length = time - start_time
                notes.append((time, length, event.pitch))
            except KeyError:
                pass
    return notes, instrument

def notes_to_time_series(notes, quantum):
    quantum_series = []
    series_end_quantum = 0
    for note in notes:
        time, length, pitch = note
        time_quantum = (time + quantum // 2) // quantum
        length_quantum = (length + quantum // 2) // quantum
        if (time_quantum < series_end_quantum):
            raise RuntimeError("overlap")
        for i in xrange(time_quantum - series_end_quantum):
            quantum_series.append(0)
            series_end_quantum += 1
        for i in xrange(length_quantum):
            quantum_series.append(pitch)
            series_end_quantum += 1
    return quantum_series

outputs = []

for midi_file_path in midi_files_iter('.'):
    pattern = midi.read_midifile(midi_file_path)
    resolution = pattern.resolution
    track_num = 0
    for track in pattern:
        track_num += 1
        #print(track)
        try:
            notes, instrument = extract_notes(track)
            #print("START:")
            #print(notes)
            if (len(notes)):
                quantum = (resolution + 3) // 6
                pitches = notes_to_time_series(notes, quantum)
                output = RCFF(pitches, instrument, midi_file_path)
                outputs.append(output)
                #pitches_bytes = array.array('B', pitches)
                #with open(midi_file_path + '_track_' + str(track_num) + '.raw', 'wb') as pickle_file:
                #    pickle_file.write(pitches_bytes)
        except RuntimeError as e:
            pass #print(e.message)
    #exit(0)
    #print pattern

for composition in outputs:
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    pattern.resolution = 480
    
    #track0 = midi.Track()
    #track0.make_ticks_rel()
    #pattern.append(track0)
    ##event = midi.SmpteOffsetEvent(tick=0, data=[96, 0, 0, 0, 0])
    ##track0.append(event)
    #event = midi.TimeSignatureEvent(tick=0, data=[4, 2, 24, 8])
    #track0.append(event)
    #event = midi.SetTempoEvent(tick=0, data=[7, 129, 27, 143])
    #track0.append(event)
    #event = midi.EndOfTrackEvent()
    #track0.append(event)
    
    
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    track.make_ticks_rel()
    # Append the track to the pattern
    pattern.append(track)

    if composition.instrument < 0:
        composition.instrument = 0

    #event = midi.ProgramChangeEvent(data=[composition.instrument])
    #track.append(event)

    #event = midi.ControlChangeEvent(control=7, value=100)
    #track.append(event)

    
    #event = midi.ControlChangeEvent(control=10, value=90)
    #track.append(event)

    quantum = -1
    for pitch in pitches:
        quantum += 1
        # Instantiate a MIDI note on event, append it to the track
        on = midi.NoteOnEvent(tick=0, velocity=20, pitch=midi.C_3)
        track.append(on)
        # Instantiate a MIDI note off event, append it to the track
        off = midi.NoteOffEvent(tick=100, pitch=midi.C_3)
        track.append(off)
    
    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Save the pattern to disk
    midi.write_midifile(composition.midi_file_path + '_track_' + str(track_num) + '.mid', pattern)
