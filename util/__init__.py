import os

_label2id = {
    "отказ": 0,
    "отмена": 1,
    "подтверждение": 2,
    "начать осаживание": 3,
    "осадить на (количество) вагон": 4,
    "продолжаем осаживание": 5,
    "зарядка тормозной магистрали": 6,
    "вышел из межвагонного пространства": 7,
    "продолжаем роспуск": 8,
    "растянуть автосцепки": 9,
    "протянуть на (количество) вагон": 10,
    "отцепка": 11,
    "назад на башмак": 12,
    "захожу в межвагонное,пространство": 13,
    "остановка": 14,
    "вперед на башмак": 15,
    "сжать автосцепки": 16,
    "назад с башмака": 17,
    "тише": 18,
    "вперед с башмака": 19,
    "прекратить зарядку тормозной магистрали": 20,
    "тормозить": 21,
    "отпустить": 22,
}

_id2label = {
    0: "отказ",
    1: "отмена",
    2: "подтверждение",
    3: "начать осаживание",
    4: "осадить на (количество) вагон",
    5: "продолжаем осаживание",
    6: "зарядка тормозной магистрали",
    7: "вышел из межвагонного пространства",
    8: "продолжаем роспуск",
    9: "растянуть автосцепки",
    10: "протянуть на (количество) вагон",
    11: "отцепка",
    12: "назад на башмак",
    13: "захожу в межвагонное,пространство",
    14: "остановка",
    15: "вперед на башмак",
    16: "сжать автосцепки",
    17: "назад с башмака",
    18: "тише",
    19: "вперед с башмака",
    20: "прекратить зарядку тормозной магистрали",
    21: "тормозить",
    22: "отпустить",
}

def label2id(label: str) -> int:
    global _label2id
    return _label2id[label]


def id2label(id: int) -> str:
    global _id2label
    return _id2label[id]

def get_labels():
    return list(_label2id.keys())

def get_path(*args):
    """Return the path to a file in the main directory."""
<<<<<<< HEAD
    # Check that the arguments are not empty
    if not args:
        raise ValueError("At least one argument is required")

    # Check that the arguments are strings
    if not all(isinstance(arg, str) for arg in args):
        raise ValueError("All arguments must be strings")

    path = os.path.join(os.path.dirname(__file__), "..", *args)

    return path
=======
    if not args or not all(isinstance(arg, str) for arg in args):
        raise ValueError("At least one string argument is required")
    
    return os.path.join(os.path.dirname(__file__)[:-4], *args)
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
