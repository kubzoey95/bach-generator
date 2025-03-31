import asyncio
import time

from mingus.midi import fluidsynth


async def play_sound(pitch, middle_pitch):
    fluidsynth.play_Note(pitch + middle_pitch, 0, 127)
    await asyncio.sleep(0.5)
    fluidsynth.stop_Note(pitch + middle_pitch, 0)
    # await asyncio.sleep(1)
    # fs.noteoff(0, pitch + middle_pitch)


async def play(seq, rest_divisor=512, middle_pitch=60):
    fluidsynth.init("Piano_Infinity_Soundfont.sf2", "pulseaudio")
    pitch = 0
    last_event = float("-inf")
    for s in seq:
        cmd, num = s.split("_")
        num = int(num)
        if cmd == "PITCH":
            pitch += num
        if cmd == "PLAY":
            asyncio.create_task(play_sound(pitch, middle_pitch))
            print("played", pitch + middle_pitch)
        if cmd == "REST":
            to_wait = num / rest_divisor
            while real_to_wait := (time.time() - last_event) < to_wait:
                await asyncio.sleep(max(0.001, 0.8 * (to_wait - real_to_wait)))
            last_event = time.time()
            print("rested", to_wait)
