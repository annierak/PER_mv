import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_max_area_contour(contour_list):
    """
    Given a list of contours finds the contour with the maximum area and
    returns
    """
    contour_areas = np.array([cv2.contourArea(c) for c in contour_list])
    max_area = contour_areas.max()
    max_ind = contour_areas.argmax()
    max_contour = contour_list[max_ind]
    return max_contour, max_area

def order_by_area(contour_list):
    """
    Given a list of contours finds the contour with the maximum area and
    returns
    """
    contour_areas = np.array([cv2.contourArea(c) for c in contour_list])
    indices = list(reversed(np.argsort(contour_areas)))
    reordered_list = [contour_list[index] for index in indices]
    return reordered_list

fps = 30.
time_per_frame = int(np.ceil(1000./fps))

def show_image(img,size=800):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', size, size)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_frame(img,size=800,time_per_frame=time_per_frame):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', size,size)
    cv2.imshow('image',img)
    cv2.waitKey(time_per_frame)
    cv2.destroyAllWindows()

threshold = 70

vidcap = cv2.VideoCapture('NPF_2.avi')
success,image = vidcap.read()

# plt.ion()
# plt.figure()
show_frame(image)

# while success:
#     success,image = vidcap.read()
#     show_image(image)

while success:
    success,image = vidcap.read()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rval, threshold_image = cv2.threshold(image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY_INV)
    threshold_image = cv2.adaptiveThreshold(image, np.iinfo(image.dtype).max,\
     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,251,2)
    contour_list, _ = cv2.findContours(threshold_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    max_contour, max_area = get_max_area_contour(contour_list)
    contour_list = order_by_area(contour_list)
    contour_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image,contour_list[0],-1,(0,0,255),2)
    # cv2.drawContours(contour_image,contour_list[1:3],-1,(0,0,255),2)
    # display = contour_image
    # plt.draw()
    show_image(contour_image)
