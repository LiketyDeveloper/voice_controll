from ai.nltk_utils import get_number_from_text

def perform_action(command: str, text):
    
    match command:
        case "move_forward":
            return  "Едем вперед"

        case "move_backwards":
            return "Едем назад"
        
        case "increase_speed":
            add_speed = get_number_from_text(text)
            return f"Ускоряемся на {add_speed}"
        
        case "decrease_speed":
            sub_speed = get_number_from_text(text)
            return f"Замедляемся на {sub_speed}"
        
        case "set_speed":
            new_speed = get_number_from_text(text)
            return f"Устанавливаем скорость на {new_speed}"
        
        case "stop":
            return "Останавливаемся"
        
        case _:
            return "Непонятная команда"
