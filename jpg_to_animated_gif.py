import os
import imageio

filenames = []
for file in os.listdir():
    print(file)
    if file.split(".")[-1] == "jpg":
        filenames.append(file)

with imageio.get_writer("./movie.gif", mode="I") as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
