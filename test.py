import asyncio

class Timer:
    def __init__(self, name, interval):
        self.name = name
        self.interval = interval

async def timer_task(timer):
    print(f"Timer {timer.name} started")
    await asyncio.sleep(timer.interval)
    print(f"Timer {timer.name} is ready")

async def main():
    timers = {}

    while True:
        user_input = await asyncio.to_thread(input, "Enter 'timer' to start a timer, 'list' to list timers, or 'exit' to exit: ")
        if user_input.lower() == "timer":
            timer_name = await asyncio.to_thread(input, "Enter a name for the timer: ")
            if timer_name in timers:
                print("Timer with this name already exists. Please choose a different name.")
                continue
            interval = int(await asyncio.to_thread(input, "Enter the timer interval in seconds: "))
            timer = Timer(timer_name, interval)
            timers[timer_name] = asyncio.create_task(timer_task(timer))
        elif user_input.lower() == "list":
            print("Active timers:")
            for timer in timers:
                print(timer)
        elif user_input.lower() == "exit":
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())
