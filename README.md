##  Image Processing Exercise 1

#### Python Version and Platform:
This program has been tested on Python 3.10 on a Windows 10 machine.

This project is part of the Image Processing and Computer Vision course, and its objective is to perform different image processing techniques.

The project includes four main tasks, which are:

1. Reading and displaying an image provided
2. Converting an image between two color spaces: RGB and YIQ
3. Performing Histogram Equalization on images
4. Performing Image Quantization and Gamma Correction on images

#### The files included in this project are:

* ex1_main.py: The main file provided in the assignment
* ex1_utils.py: The file containing the functions for the tasks mentioned above
* gamma.py: The file containing the Gamma Correction function
* Ex1.pdf: The instructions file for this assignment
* README.md: This file you are reading now
* images: A folder containing the images provided and additional ones that I added for testing

#### Functions:

* myID() - Returns the ID of the user.

* imReadAndConvert(filename: str, representation: int) -> np.ndarray -
Reads an image from a given path and converts it to grayscale or RGB format, as specified by the representation
parameter.
Returns the image as a ndarray.

* imDisplay(filename: str, representation: int) -
Reads an image and displays it in grayscale or RGB format, as specified by the representation parameter.

* transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray -
Converts an RGB image to YIQ color space.

* transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray -
Converts a YIQ image to RGB color space.

* hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray) -
Performs histogram equalization on an image and returns the equalized image, the original histogram, and the equalized
histogram.

* gammaDisplay(img_path: str, rep: int) -> None -
Provides a GUI for adjusting the gamma value of an image.
It takes the path to the image and create a window where the user can adjust the gamma value through a trackbar.
The function applies gamma correction to the image based on the user's input and displays the corrected image in the window.



