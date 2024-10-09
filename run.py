import multiprocessing.process
import os

# from speech_recognition import listen
from ai.model import load_model
from loguru import logger
from transport import Train
import multiprocessing as mp



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

        train.current_state = command_name
        
    
if __name__ == "__main__":
    # p = mp.Process(target=os.system, args=("python manage_states.py",))
    # p.start()
    
    main()
