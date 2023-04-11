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
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    window_size: tuple = (800, 600)
    # Read the image
    if rep == 1:  # Gray_Scale representation
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:  # =2, read as BGR image
        img = cv2.imread(img_path)

    # Define the function to be called when the trackbar is moved
    def on_gamma_change(gamma):
        # Convert the integer value to a float in the range [0, 2]
        gamma = gamma / 100.0
        # Apply gamma correction to the image
        corrected_img = np.power(img / 255.0, gamma)
        corrected_img = np.uint8(corrected_img * 255)
        # Show the corrected image
        cv2.imshow("Gamma Correction", corrected_img)

    # Create a window to display the image and trackbar
    cv2.namedWindow("Gamma Correction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gamma Correction", window_size[0], window_size[1])

    # Create a trackbar with values from 0 to 200
    # and set the initial value to 100 (gamma = 1.0)
    cv2.createTrackbar("Gamma", "Gamma Correction", 100, 200, on_gamma_change)

    # Show the initial image
    cv2.imshow("Gamma Correction", img)

    # Wait for a key press and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)
    gammaDisplay('beach.jpg', LOAD_GRAY_SCALE)
    gammaDisplay('dark.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
