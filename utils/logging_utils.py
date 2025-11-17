
def show_progress_bar(progress: int, total: int, title: str = "Progress", width: int = 20):
    percent_complete = round(progress / total, 2)
    proportional_progress = round(percent_complete * width)
    print(title + ": [" + (proportional_progress * "#") + ((width - proportional_progress) * " ") + "] " + str(percent_complete * 100) + "%", end=" "*(width*2) + "\r")