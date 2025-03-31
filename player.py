import numpy
import pygame
import time
from functools import lru_cache
from scipy.signal import sawtooth
import numpy as np

@lru_cache(256)
def create_wave(pitch, baseFreq=220):
    sampleRate = 44100
    freq = baseFreq * pow(2, pitch / 12)

    arr = (2048 * sawtooth(2.0 * numpy.pi * freq * numpy.arange(0, int(sampleRate * 0.1)) / sampleRate)).astype(numpy.int16)
    # arr = numpy.array([ for x in range(0, sampleRate)]).astype(numpy.int16)
    arr2 = numpy.c_[arr, arr]
    sound = pygame.sndarray.make_sound(arr2)
    return sound


def play(seq, rest_divisor=4, baseFreq=220):
    pygame.mixer.init(44100, -16, 2, 512)
    
    pitch = 0
    last_event = float("-inf")
    for s in seq:
        cmd, num = s.split("_")
        num = int(num)
        if cmd == "PITCH":
            pitch += num
        if cmd == "PLAY":
            create_wave(pitch, baseFreq).play(maxtime=100, fade_ms=50)
        if cmd == "REST":
            to_wait = num / rest_divisor
            while time.time() - last_event < to_wait:
                time.sleep(0.001)
            last_event = time.time()
