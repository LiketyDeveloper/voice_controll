from util import get_path

"""Command Identifier Neural Netword Settings"""

MODEL_FILE_PATH = get_path("ai", "data", "model.pth")
<<<<<<< HEAD
DATASET_FILE_PATH = get_path("ai", "data", "dataset.json")
VOCABULARY_FILE_PATH = get_path("ai", "data", "vocabulary.json")
NN_TRAIN_EPOCHS = 300
=======
VOCAB_PATH = get_path("ai", "data", "vocabulary.json")
ANNOTATIONS_PATH = get_path("ai", "dataset", "annotation")
NN_TRAIN_EPOCHS = 100
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
