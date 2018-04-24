import numpy as np
import cv2
import matplotlib.pyplot as plt


def order_by_area(contour_list):
    """
    Given a list of contours finds the contour with the maximum area and
    returns
    """
    contour_areas = np.array([cv2.contourArea(c) for c in contour_list])
    indices = list(reversed(np.argsort(contour_areas)))
    reordered_list = [contour_list[index] for index in indices]
    return reordered_list



threshold = 70

vidcap = cv2.VideoCapture('NPF_2.avi')
success,image = vidcap.read()

plt.ion()
plt.figure(1)
plotted_image = plt.imshow(image)
# show_frame(image)
# plot_image(image)

# while success:
#     success,image = vidcap.read()
#     print 'here'
#     plotted_image.set_data(image)
#     plt.draw()
#     plt.pause(1)
while success:
    success,image = vidcap.read()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rval, threshold_image = cv2.threshold(image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY_INV)
    threshold_image = cv2.adaptiveThreshold(image, np.iinfo(image.dtype).max,\
     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,251,2)
    contour_list, _ = cv2.findContours(threshold_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour_list = order_by_area(contour_list)
    contour_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image,contour_list[1:3],-1,(0,0,255),2)
    plotted_image.set_data(image)
    plt.draw()
    plt.pause(.1)
    # raw_input('  ')
