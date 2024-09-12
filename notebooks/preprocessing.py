import numpy as np
import cv2

def extract_red_channel(image: np.ndarray) -> np.ndarray:
    _, _, r = cv2.split(image)
    return r

def clahe(image: np.ndarray) -> np.ndarray:
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9,9))
    equalized = cl.apply(image)
    return equalized

def bilateral_filter(image: np.ndarray) -> np.ndarray:
    bil_fil = cv2.bilateralFilter(src=image, d=5, sigmaColor=50, sigmaSpace=50)
    return bil_fil

def pipeline(image: np.ndarray) -> np.ndarray:
    image = extract_red_channel(image)
    image = clahe(image)
    image = bilateral_filter(image)
    return image
