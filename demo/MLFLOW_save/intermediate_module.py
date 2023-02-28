# we import the model as an alias to showcase that cloudflow still work 
from sub_folder.subsub_folder import is_sub

def func_print(string):
    """
    Prints the given string and calls the 'test' function from the 'is_sub' module in the 'subsub_folder' package.

    Args:
    - string (str): the string to print

    Returns:
    - None
    """
    print("print from intermediate_module --> {}".format(string))
    is_sub.get_print_is_sub()