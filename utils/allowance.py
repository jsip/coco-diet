import os
import numpy as np
import sounddevice as sd
import random

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.schedule import Schedule

def play_noise(duration=3, volume=0.35, sample_rate=10000):
    "duration in seconds, volume 0-1, sample_rate in hz"
    
    noise = np.random.uniform(-1, 1, int(sample_rate * duration)) * random.uniform(0, volume)

    try:
        sd.play(noise, samplerate=sample_rate)
        sd.wait()
        print(f"Played {duration}s of noise.")
    except Exception as e:
        print(f"An error occurred while playing noise: {e}")


def check_schedule(cat):
    if cat == "rico":
        return True

    if Schedule.can_eat():
        return True

    play_noise()