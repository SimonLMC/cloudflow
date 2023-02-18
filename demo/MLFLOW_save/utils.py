import requests
import intermediate_module

def get_image(url = "oui"):
    intermediate_module.func_print("lancement get_image")
    return requests.get(url, stream=True).raw