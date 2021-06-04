import cv2
import matplotlib.pyplot as plt
import numpy as np

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))

def toGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise(image, value):
    return cv2.GaussianBlur(image,(value, value), 0)
 
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def blackhat(image, kernelSize = None):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize) if kernelSize else rectKernel
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

def whiteSquarify(image):
    gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    # p = int(image.shape[1] * 0.05)
    # thresh[:, 0:p] = 0
    # thresh[:, image.shape[1] - p:] = 0
    return thresh

def showImage(image, gray=False):
    if gray:
        plt.imshow(image, cmap = plt.cm.gray)
    else:
        plt.imshow(image)
    plt.show()