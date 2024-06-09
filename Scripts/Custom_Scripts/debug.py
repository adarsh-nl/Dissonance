DEBUG = True

def debug(*args):

    """
    Display debug information.

    This function prints the provided arguments to the console and writes them to
    the Streamlit app if in debug mode.

    Parameters:
    - *args: Variable number of arguments to be printed and written.

    Returns:
    - None
    """

    if DEBUG:
        print(*args)