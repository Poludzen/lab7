import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('parrots.jpg')
plt.imshow(img[...,::-1]);
plt.title("main image(to show next image close this)")
plt.show()

# function that use median filder
def median_filter(img, ksize):
    result = cv2.medianBlur(img, ksize)
    return result


filtered = median_filter(img, 11)
plt.title("After Median Filter(to show next image close this)")
plt.imshow(filtered[...,::-1]);
plt.show()


def gauss_kernel(ksize = 5, sigma = 2.5):
    """ returns gaussian kernel size of k and variance -  sigma """
    # ksize - kernel size
    # sigma - variance(width of a filter)
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    e = np.float32((xx**2 + yy**2) / (2. * sigma**2))
    kernel = 1. / (sigma * np.sqrt(2. * np.pi)) * np.exp(-e)
    return kernel / np.sum(kernel)


# using gauss filter
result = cv2.filter2D(img, cv2.CV_32F, gauss_kernel(8,10))
# to make picture bigger
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
plt.title("After Gauss filter(to show next image close this)")
ax.imshow(np.uint8(result[...,::-1]))
plt.show()


# gray scaled image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.title("Gray img")
plt.imshow(gray_image,cmap='gray')
plt.show()


# using equation of histograms
equ = cv2.equalizeHist(gray_image)
plt.title("after equalize hist filter(to show next image close this)")
plt.imshow(equ,cmap='gray')
plt.show()
