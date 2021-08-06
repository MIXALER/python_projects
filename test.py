import numpy as np
import cv2

def show_random_image():
    img = np.random.random([256, 256, 3])
    cv2.imshow('test', img)
    cv2.waitKey(100000)


if __name__ == "__main__":
    show_random_image()
