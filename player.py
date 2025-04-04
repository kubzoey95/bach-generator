import asyncio
import time
import random
from mingus.midi import fluidsynth
from collections import deque

async def play_sound(pitch, middle_pitch):
    ch = random.randint(0, 16)
    fluidsynth.play_Note(pitch + middle_pitch, ch, 127)
    await asyncio.sleep(0.5)
    fluidsynth.stop_Note(pitch + middle_pitch, ch)
    # await asyncio.sleep(1)
    # fs.noteoff(0, pitch + middle_pitch)


async def play(seq, rest_divisor=50, middle_pitch=60):
    fluidsynth.init("[GD] The Grandeur D.sf2", "pulseaudio")
    pitch = 0
    last_event = float("-inf")
    events = deque(maxlen=5)
    for s in seq:
        cmd, num = s.split("_")
        num = int(num)
        if cmd == "PITCH":
            pitch += num
        if cmd == "PLAY":
            asyncio.create_task(play_sound(pitch, middle_pitch))
            events.append(("played", pitch + middle_pitch))
        if cmd == "REST":
            to_wait = num / rest_divisor
            while real_to_wait := (time.time() - last_event) < to_wait:
                await asyncio.sleep(max(0.001, 0.8 * (to_wait - real_to_wait)))
            last_event = time.time()
            events.append(("rested", to_wait))
            events.append(("real rested cs", num))
        # for e in events:
        #     print(e, end="\r", flush=True)