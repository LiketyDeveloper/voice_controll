import json, pyaudio, os
from vosk import Model, KaldiRecognizer
from vosk import SetLogLevel
from pydub import AudioSegment

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

def recognize_speech_from_file(file_path):
    """
    Recognize speech from an audio file using vosk KaldiRecognizer
    """
    recognizer = KaldiRecognizer(model, 16000)

# Open the audio file
    wf = wave.open(file_path, "rb")

    # Process the audio file
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            print(result)
        else:
            partial_result = recognizer.PartialResult()
            print(partial_result)

    # Get the final recognized result
    result = recognizer.FinalResult()
    return result
