import datetime

def get_print_with_time(message):
    """
    Logs a message to the console with a timestamp and formatted text.

    Args:
        message (str): The message to log.

    Returns:
        None.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print("=" * len(formatted_message))
    print(formatted_message)
    print("=" * len(formatted_message))
