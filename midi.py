from midiutil import MIDIFile
import numpy as np

class MidiConverter:
    def convert_to_midi(self, piece: np.ndarray, name, resolution=1/16):
        midi_file = MIDIFile(1)
        midi_file.addTempo(0, 0, 60)
        # keeps track of previous note to know if note is held
        previous_notes = list(piece[0])
        # keeps track of the duration the note has been held
        held_note_duration = [0] * len(piece[0])

        for t, notes in enumerate(piece):
            for i, note in enumerate(notes):
                if previous_notes[i] == note:
                    held_note_duration[i] += resolution
                else: 
                    prev_note = previous_notes[i]
                    # silent notes are nan
                    if not(np.isnan(prev_note)):
                        time = t*resolution-held_note_duration[i]
                        midi_file.addNote(0,i,int(prev_note),time,held_note_duration[i],100)
                    # reset if new note
                    held_note_duration[i] = resolution
                    previous_notes[i] = note
        
        with open(f"{name}.mid", "wb") as output_file:
            midi_file.writeFile(output_file)

