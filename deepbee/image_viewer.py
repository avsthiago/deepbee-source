import os
from argparse import ArgumentParser
import cv2

# creates the argument parser and use it for parsing the arguments
ap = ArgumentParser()
ap.add_argument('-i', '--image-folder', required=True,
                help='folder where the downloaded images are')
args = vars(ap.parse_args())
image_folder = args['image_folder']

# creates a window that will be used as a placeholder for the images
cv2.namedWindow('image_view', cv2.WINDOW_NORMAL)

# iterates over each image and shows it
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    cv2.imshow('image_view', image)
    cv2.waitKey(0)
