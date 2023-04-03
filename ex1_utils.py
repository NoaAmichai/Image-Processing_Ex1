"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""

from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int64:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206996381


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE(1) or RGB(2)
    :return: The image object
    """
    # read the image with OpenCV
    img = cv2.imread(filename)

    # check if the image was successfully loaded
    if img is None:
        raise ValueError(f"Error: Could not read image file {filename}")

    # convert the image to GRAY_SCALE if requested
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert the image to RGB if requested
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Error: Unsupported image representation")

    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    elif img.dtype == np.float32 or img.dtype == np.float64:
        img = img.astype(np.float64)
    else:
        raise TypeError("Invalid image type")

    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    try:
        img = imReadAndConvert(filename, representation)
        plt.imshow(img, cmap='gray' if representation == 1 else None)
        plt.show()
    except Exception as e:
        print(f"Error: {str(e)}")
    # plt.figure()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])

    return np.dot(imgRGB, yiq_from_rgb)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    rgb_from_yiq = np.array([[1.0, 0.956, 0.619],
                             [1.0, -0.272, -0.647],
                             [1.0, -1.106, 1.703]])

    return np.dot(imgYIQ, rgb_from_yiq)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
