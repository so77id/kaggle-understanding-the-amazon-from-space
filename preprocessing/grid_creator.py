import Image
from os import listdir
from os.path import isfile, join
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder_path', help='Folder with images', required=True)
parser.add_argument('-s1', '--grid_size1', help='GRID number of lines', type=int, required=True)
parser.add_argument('-s2', '--grid_size2', help='GRID number of columns', type=int, required=True)
ARGS = parser.parse_args()
type='int'

files = [f for f in listdir(ARGS.folder_path)]
files = sorted(files)

grid_pixels1 = ARGS.grid_size1 * 256
grid_pixels2 = ARGS.grid_size2 * 256

new_im = Image.new('RGB', (grid_pixels2, grid_pixels1))

index = 0
for i in range(0, grid_pixels1, 256):
    for j in range(0, grid_pixels2, 256):
        im = Image.open(ARGS.folder_path + '/' + files[index])
        im.thumbnail((256, 256))
        new_im.paste(im, (j, i))
        index += 1

new_im.save("newimg.png")