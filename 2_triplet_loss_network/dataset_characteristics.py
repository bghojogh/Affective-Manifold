def get_image_height():
    # image_height = 32
    image_height = 28
    return image_height

def get_image_width():
    # image_width = 32
    image_width = 28
    return image_width

def get_image_n_channels():
    # n_channels = 3
    n_channels = 1
    return n_channels

def get_class_names():
    # class_names = ["TUMOR", "STROMA", "MUCUS", "LYMPHO", "DEBRIS", "SMOOTH MUSCLE", "ADIPOSE", "BACKGROUND", "NORMAL"]
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return class_names