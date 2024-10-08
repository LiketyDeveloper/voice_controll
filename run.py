from speech_recognition import listen
from loguru import logger

if __name__ == "__main__":
    logger.success("Application started")
    for text in listen():         
        print(text)