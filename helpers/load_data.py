import os
import requests


SOURCE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_data(output_dir: str = "data/input.txt") -> str:
    """
    Loads data from a specified source and saves it to a file.

    Args:
        output_dir (str): The directory where the data will be saved. Defaults to "data/input.txt".

    Returns:
        str: The loaded data as a string.
    """
    if not os.path.exists(output_dir):
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        response = requests.get(SOURCE)
        with open(output_dir, "w", encoding="utf-8") as f:
            f.write(response.text)
    
    with open(output_dir, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text