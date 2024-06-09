import re

def create_clientid(client_name):
    """
    Create a client ID from the given client name using regular expression matching.

    Parameters:
    - client_name (str): Name of the client containing numeric characters.

    Returns:
    - int: Extracted client ID from the client name.

    Example:
    ```
    client_name = "client123"
    client_id = create_clientid(client_name)
    ```

    The function uses a regular expression to search for numeric characters within the
    provided client name. If a match is found, it extracts the numeric characters and
    converts them to an integer, returning the resulting client ID.
    """
    match = re.search(r'\d+', client_name)
    if match:
        client_id = int(match.group())
        return client_id
