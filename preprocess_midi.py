import mido
from tqdm import tqdm
from pathlib import Path


def merge_midi_tracks(input_midi):
    midi = None
    try:
        midi = mido.MidiFile(input_midi)
    except Exception as e:
        print(e)
    if midi:
        merged_midi = mido.MidiFile()
        assert len(merged_midi.tracks) == 0
        merged_midi.tracks.append(mido.merge_tracks(midi.tracks))
        assert len(merged_midi.tracks) == 1
        merged_midi.save(input_midi)


for p in tqdm(Path("baroque").glob("**/*")):
    merge_midi_tracks(p)