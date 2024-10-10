# from speech_recognition import listen
from ai.model import load_model
from loguru import logger
from transport import Train
from threading import Thread



def main():
    model = load_model()
    logger.success("Application started")
    
    train_id = int(input("Введите id поезда: "))
    train = Train(train_id)
    # for text in listen():
    while True:
        text = input("Введите команду: ")
        
        if text == "выйти":
            logger.info("Вы вышли из программы")         
        
        logger.info(f"Вы сказали: {text}")
        command_name = model.invoke(text)
        
        Thread(target=train.perform_action, args=(command_name, text)).start()     
    

if __name__ == "__main__":
    main()
