import json
with open('config.json') as f:
    params = json.load(f)

def get_image_height():
    image_height = params["image_size"][0]
    return image_height

def get_image_width():
    image_width = params["image_size"][1]
    return image_width

def get_image_n_channels():
    n_channels = params["image_size"][2]
    return n_channels

def get_class_names():
    n_classes = params["n_classes"]
    class_names = [str(i) for i in range(n_classes)]
    return class_names