from miditok import TokenizerConfig, Structured
from symusic import Score
from pathlib import Path
import json
import numpy as np
import random
from tqdm import tqdm

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(use_chords=0, use_programs=False, use_velocities=0, use_rests=1, use_tempos=0, 
                         use_pitchdrum_tokens=0, use_pitch_bends=0, use_pitch_intervals=0, use_sustain_pedals=0, use_time_signatures=0)
tokenizer = Structured(config)



for p in tqdm(Path("baroque").glob("**/*")):
    if not p.is_file():
        continue
    try:
        midi = Score(p)
    except Exception as e:
        print(str(e))
        continue
    tokens = tokenizer(midi)
    converted_back_midi = tokenizer.decode(tokens).sort()
    # big_track = converted_back_midi.tracks[0]
    
    # for track in converted_back_midi.tracks[1:]:
    #     big_track.notes.extend(track.notes)
    # converted_back_midi.tracks = [big_track]
    # converted_back_midi.sort()
    # notes = converted_back_midi.tracks[0].notes
    
    # end = converted_back_midi.tracks[0].end()
    
    for i, track in enumerate(converted_back_midi.tracks):
        out_path = Path("baroque_processed", *p.parts[1:], f"track_{i}").with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        track.sort()
        
        out = [[n.pitch, n.start] for n in track.notes]
        out.append(["END", track.end()])
        with out_path.open("w") as f:
            json.dump(out, f)
    
    # out = np.array([[n0.pitch, n1.start - n0.start] for n0, n1 in zip(notes[:-1], notes[1:])])
    # notes.sort(lambda x: (x.start, random.randint(-99999, 99999)))
    
    # out = np.array([[n0.pitch, n1.start - n0.start] for n0, n1 in zip(notes[:-1], notes[1:])])
    
    # out[1:, 0] = out[1:, 0] - out[:-1, 0]
    # out[:, 0] = np.sign(out[:, 0]) * (np.abs(out[:, 0]) % 12)
    
