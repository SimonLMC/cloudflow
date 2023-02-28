from PIL import Image
import subprocess
import utils
import get_print
import pandas as pd
import numpy as np
from sklearn import ensemble

def download_image(url):
    """
    Downloads an image from a given URL and returns it as a PIL Image object.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image: The downloaded image as a PIL Image object.
    """
    # Download the image from the given URL
    image_data = utils.get_image(url)
    
    # Return the image as a PIL Image object
    return Image.open(image_data)

def download_story():
    """
    Downloads a random short story and returns it as a string.

    Returns:
        str: A random short story as a string.
    """
    get_print.get_print_with_time("Launch download_story")
    
    # Retrieve a random short story using the provided API endpoint
    return eval(subprocess.Popen("curl https://shortstories-api.onrender.com/", stdout=subprocess.PIPE, shell=True).communicate()[0])["story"]