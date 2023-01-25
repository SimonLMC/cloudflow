import requests
import no_import

def get_image(url = "oui"):
    no_import.not_to_import("lancement get_image")
    return requests.get(url, stream=True).raw


def get_print(x):
    print(x) 