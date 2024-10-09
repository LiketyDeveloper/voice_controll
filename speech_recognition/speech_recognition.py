import json, pyaudio, os
from vosk import Model, KaldiRecognizer
from vosk import SetLogLevel

# Disabling defult vosk logs
SetLogLevel(-1)

# Loading the model from file
model = Model(os.path.join(os.path.dirname(__file__), "sr_model"))

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