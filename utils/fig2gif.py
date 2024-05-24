import imageio
import os
import sys

# Path to the directory where your images are stored
N = sys.argv[1]

path_to_images = f"N={N}"

# Collecting all image filenames
def get_epoch(filename):
    base = os.path.basename(filename)  # Remove directory if present
    name, ext = os.path.splitext(base)  # Separate the extension
    try:
        return int(name.split('_')[-1].split('.')[0].split('=')[-1])  # Assumes format is "name_epoch.ext"
    except ValueError:
        return 0  # or handle appropriately if some filenames are formatted unexpectedly

# Collecting all image filenames and sorting by epoch number
str_n = f"N={N}_"
file_lists = os.listdir()
for path_name in os.listdir():
    if str_n in path_name:
        path_to_images = path_name
# print(path_to_images)
# assert 0
images = sorted([img for img in os.listdir(path_to_images) if img.endswith(".png")], key=get_epoch)
# images = sorted([img for img in os.listdir(path_to_images) if img.endswith(".png")])
# print(images[:10])
print(len(images))
# Creating a list of images for the animation
frames = [imageio.imread(os.path.join(path_to_images, images[i])) for i in range(0, 500, 10)]

# # Save the frames as an animated GIF
imageio.mimsave(f"output_N={N}.gif", frames, duration=0.1)  # Adjust duration as needed