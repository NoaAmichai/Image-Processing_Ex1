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
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

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


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])

    return np.dot(imgRGB, yiq_from_rgb.T)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    rgb_from_yiq = np.array([[1.0, 0.956, 0.619],
                             [1.0, -0.272, -0.647],
                             [1.0, -1.106, 1.703]])

    return np.dot(imgYIQ, rgb_from_yiq.T)


def hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """

    # Check if the image is grayscale or RGB
    if len(imOrig.shape) == 2:  # Grayscale
        img = imOrig * 255
        # Compute histogram of grayscale image
        histOrig, bins = np.histogram(img.flatten(), 256, [0, 255])
    else:  # RGB
        # Convert RGB image to YIQ color space
        img = transformRGB2YIQ(imOrig)
        img[:, :, 0] = img[:, :, 0] * 255
        # Compute histogram of Y channel in YIQ image
        histOrig, bins = np.histogram(img[:, :, 0].flatten(), 256, [0, 255])

    # Compute the cumulative distribution function (cdf)
    cdf = histOrig.cumsum()

    # Normalize the cdf to a 0-255 scale
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Create a LookUpTable(LUT)
    LUT = np.round(cdf).astype('uint8')

    # Replace each intensity i with LUT[i]
    imgEq = LUT[img.astype('uint8')]

    # Compute the histogram of the equalized image
    histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    # Normalize image values back to [0,1] range
    if len(imOrig.shape) == 2:  # Grayscale
        imgEq = imgEq / 255
    else:  # RGB
        img[:, :, 0] = imgEq[:, :, 0] / 255
        imgEq = transformYIQ2RGB(img)

    return imgEq, histOrig, histEq


def case_RGB(imgOrig: np.ndarray) -> (bool, np.ndarray, np.ndarray):
    isRGB = imgOrig.ndim == 3 and imgOrig.shape[-1] == 3
    if isRGB:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = imgYIQ[..., 0]
        return True, imgYIQ, imgOrig
    return False, None, imgOrig


def back_to_rgb(yiq_img: np.ndarray, y_to_update: np.ndarray) -> np.ndarray:
    yiq_img[:, :, 0] = y_to_update
    return transformYIQ2RGB(yiq_img)


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    isRGB, yiq_img, imOrig = case_RGB(imOrig)

    if np.amax(imOrig) <= 1:  # picture is normalize
        imOrig = imOrig * 255
    imOrig = imOrig.astype('uint8')

    histOrg, bin_edges = np.histogram(imOrig, 256, [0, 255])

    z = np.arange(0, 256, int(255 / nQuant))  # boundaries
    z[nQuant] = 255
    q = np.zeros(nQuant)

    qImage_list = list()
    error_list = list()

    for i in range(nIter):
        new_img = np.zeros(imOrig.shape)

        for cell in range(len(q)):
            # Determine the range of pixel intensities for this cell
            left = z[cell]
            right = z[cell + 1] if cell < len(q) - 1 else 256
            cell_range = np.arange(left, right)

            # Compute the average intensity for this cell, weighted by the pixel counts in its range
            hist_cell = histOrg[left:right]
            weights = hist_cell / np.sum(hist_cell)
            q[cell] = np.sum(weights * cell_range)

            # Assign the average intensity to all pixels within the cell's range
            condition = np.logical_and(imOrig >= left, imOrig < right)
            new_img[condition] = q[cell]

        MSE = mean_squared_error(imOrig / 255, new_img / 255)
        error_list.append(MSE)

        if isRGB:
            new_img = back_to_rgb(yiq_img, new_img / 255)

        qImage_list.append(new_img)
        z[1:-1] = (q[:-1] + q[1:]) / 2
        if len(error_list) >= 2 and abs(error_list[-1] - error_list[-2]) <= sys.float_info.epsilon:  # check if converge
            break

    return qImage_list, error_list
