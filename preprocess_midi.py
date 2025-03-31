import mido
from tqdm import tqdm
from pathlib import Path


def merge_midi_tracks(input_midi):
    midi = mido.MidiFile(input_midi)
    
    merged_midi = mido.MidiFile()
    merged_midi.tracks.append(mido.merge_tracks(midi.tracks))
    
    merged_midi.save(input_midi)


for p in tqdm(Path("baroque").glob("**/*")):
    try:
        merge_midi_tracks(p)
    except Exception as e:
        print(e)