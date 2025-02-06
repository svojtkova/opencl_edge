import cv2
import time
import numpy as np


# convolution of 3x3 kernel on image
def sobelOperator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            # calculate gradient and if gradient is greater than 255 there is an edge white color is assigned
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container
    pass


# main function that changes image to B&W and then applies sobel Operator
def serial_detection(filename):
    # time starting
    start_time = time.time()
    # read Image
    img = cv2.imread(filename)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert image to 8 bit
    img = cv2.convertScaleAbs(img)
    # apply sobel operator
    # Gx [1  0  -1]
    #    [2  0  -2]
    #    [1  0  -1]
    # Gy [1   2   1]
    #    [0   0   0]
    #    [-1 -2  -1]
    img = sobelOperator(img)
    # end of process
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))
    # save image
    cv2.imwrite('serial'+filename, img)


if __name__ == '__main__':
    print("Picture 500x500")
    serial_detection("500x500.png")
    print("Picture 1000x1000")
    serial_detection("1000x1000.png")
    print("Picture 1500x1500")
    serial_detection("1500x1500.png")


