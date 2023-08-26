from datetime import datetime

def add_to_log(message: str):
    """
    Prints messages to the screen, as Python print function but with the incident time.

    Args:
        message (str): The script to be printed.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def stop_model(message: str = None, exception=BaseException):
    """
    Raise an error with a message when needed.

    Args:
        message (str): Error message
        exception (error): Error message type

    Raises:
        exception: Exception containing the user-defined message
    """
    if message is not None:
        with open("Status.txt", "w") as f:
            f.write(message)
    raise exception(message)
