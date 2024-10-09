import json, pyaudio, os
from vosk import Model, KaldiRecognizer
from vosk import SetLogLevel

from config.sr import USE_BIG_SR_MODEL, SHOW_SR_LOG, BIG_MODEL_PATH, SMALL_MODEL_PATH

# Disabling defult vosk logs
if not SHOW_SR_LOG:
    SetLogLevel(-1)

# Loading the model from file
if USE_BIG_SR_MODEL:
    model = Model(BIG_MODEL_PATH)
else:
    model = Model(SMALL_MODEL_PATH)

# Starting audion stream from microphone
rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=8000
)

stream.start_stream()

def listen():
    """Listen for audio data"""
    
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data) and len(data) > 0:
            answer = json.loads(rec.Result())
            if answer["text"]:
                yield answer["text"]