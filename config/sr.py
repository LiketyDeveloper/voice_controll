from util import get_path

"""Speech Recognition Settings"""

<<<<<<< HEAD
USE_BIG_SR_MODEL = True
=======
USE_BIG_SR_MODEL = False
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
SHOW_SR_LOG = True

BIG_MODEL_PATH = get_path("speech_recognition", "sr_model_big")
SMALL_MODEL_PATH = get_path("speech_recognition", "sr_model")