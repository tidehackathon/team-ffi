def set_color(value):
    if value < 0.2:
        return "rgb(234,52,60)"
    elif value < 0.4:
        return "rgb(239,123,0)"
    elif value < 0.6:
        return "rgb(255,214,2)"
    elif value < 0.8:
        return "rgb(0,181,136)"
    else:
        return "rgb(0,135,131)"