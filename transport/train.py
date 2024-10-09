from ai.nltk_utils import get_number_from_text
from util import get_path

from loguru import logger
from transport import states

train_db = {
    1: {
        "num_of_vans": 16,
        "state": "active",
        "max_speed": 120
    },
    2: {
        "num_of_vans": 32,
        "state": "inactive",
        "max_speed": 60
    }
}

class Train:
    def __init__(self, train_id):
        if train_id not in train_db.keys():
            raise ValueError("Такого поезда не существует")
        
        if train_db[train_id]["state"] == "active":
            raise ValueError("Поезд уже в пути")
        
        logger.success(f"Вы подключились к поезду {train_id}")
        
        train_db[train_id]["state"] = "active"
        self._train_id = train_id
        self._speed = 0
        self._max_speed = train_db[train_id]["max_speed"]
        self._current_state = states.stop
       
    @property
    def train_id(self):
        return self._train_id 
       
    @property
    def speed(self):
        return self._speed
    
    @property
    def max_speed(self):
        return self._max_speed
    
    @property
    def current_state(self):
        return self._current_state
    
    @current_state.setter
    def current_state(self, new_state):
        self._current_state = new_state
        
        with open(f"{get_path("transport")}\_state_cache.txt", "w") as file:
            file.write(new_state)
    
    def perform_action(self, command: str, text: str):
    
        match command:
            case "move_forward":
                logger.info("Едем вперед")
                # time.sleep(10)
                logger.info("Timer is over")

            case "move_backwards":
                logger.info("Едем назад")
            
            case "increase_speed":
                add_speed = get_number_from_text(text)
                new_speed = self._speed + add_speed
                
                if new_speed > self.max_speed or new_speed < 0:
                    logger.info(f"Скорость должна быть в пределах от 0 до {self.max_speed}")
                else:
                    self._speed = new_speed
                logger.info(f"Ускоряемся на {add_speed}")
            
            case "decrease_speed":
                sub_speed = get_number_from_text(text)
                new_speed = self._speed - sub_speed
                
                if new_speed > self.max_speed or new_speed < 0:
                    logger.info(f"Скорость должна быть в пределах от 0 до {self.max_speed}")
                else:
                    self._speed = new_speed
                logger.info(f"Замедляемся на {sub_speed}")
            
            case "set_speed":
                new_speed = get_number_from_text(text)
                
                if new_speed > self.max_speed or new_speed < 0:
                    logger.info(f"Скорость должна быть в пределах от 0 до {self.max_speed}")
                else:
                    self._speed = new_speed
                logger.info(f"Устанавливаем скорость на {new_speed}")
            
            case "stop":
                self._max_speed = 0
                logger.info("Останавливаемся")
            
            case _:
                logger.info("Непонятная команда")
                
        logger.info(self)
            
    def __str__(self):
        return f"Поезд {self.train_id}: Текущая скорость: {self.speed}"
    