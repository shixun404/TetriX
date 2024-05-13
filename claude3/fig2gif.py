import imageio
import os
import sys

# Path to the directory where your images are stored
path_to_images = sys.argv[1]

# Collecting all image filenames
def get_epoch(filename):
    base = os.path.basename(filename)  # Remove directory if present
    name, ext = os.path.splitext(base)  # Separate the extension
    try:
        return int(name.split('^')[-1].split('.')[0].split('=')[-1])  # Assumes format is "name_epoch.ext"
    except ValueError:
        return 0  # or handle appropriately if some filenames are formatted unexpectedly


# print(path_to_images)
# assert 0
images = sorted([img for img in os.listdir(path_to_images) if img.endswith(".png")], key=get_epoch)
# images = sorted([img for img in os.listdir(path_to_images) if img.endswith(".png")])
# print(images[:10])
print(len(images))
# Creating a list of images for the animation
frames = [imageio.imread(os.path.join(path_to_images, image)) for image in images]

# # Save the frames as an animated GIF
imageio.mimsave(f"{path_to_images}.gif", frames, duration=2)  # Adjust duration as needed
