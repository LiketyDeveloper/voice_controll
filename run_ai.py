from ai.model import load_model

while True:
    model = load_model()

    user_input = input("Введите команду: ")
    print(model.invoke(user_input))