from speech_recognition import listen
from ai.model import load_model
from loguru import logger
from funcs import perform_action

if __name__ == "__main__":
    model = load_model()
    logger.success("Application started")
    
    for text in listen():
        
    # while True:
    #     text = input("Введите команду: ")
        if text == "выйти":
            logger.info("Вы вышли из программы")
            break         
        
        logger.info(f"Вы сказали: {text}")
        command_name = model.invoke(text)
        logger.info(perform_action(command_name, text))
