myID() - Returns the ID of the user (hardcoded).

imReadAndConvert(filename: str, representation: int) -> np.ndarray -
Reads an image from a given path and converts it to grayscale or RGB format, as specified by the representation
parameter.
Returns the image as a ndarray.

imDisplay(filename: str, representation: int) -
Reads an image and displays it in grayscale or RGB format, as specified by the representation parameter.

transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray -
Converts an RGB image to YIQ color space.

transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray -
Converts a YIQ image to RGB color space.

hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray) -
Performs histogram equalization on an image and returns the equalized image, the original histogram, and the equalized
histogram.