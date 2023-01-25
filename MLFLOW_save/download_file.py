from PIL import Image
import subprocess
import utils
import to_import
from sub_folder import test_import

def download_image(url):
    # Download an image with cute cats
    image_data = utils.get_image(url)
    
    return Image.open(image_data)

def download_story():
    utils.get_print("Lancement download_story")
    return eval(subprocess.Popen("curl https://shortstories-api.onrender.com/", stdout=subprocess.PIPE, shell=True).communicate()[0])["story"]