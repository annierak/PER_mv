import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats
import sys


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

def get_rightmost_point(cnt):
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    return rightmost
def get_leftmost_point(cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    return leftmost

def get_vertical_border_endpoints(x_coord,image):
    y_top = np.shape(image)[0]
    return (x_coord,0),(x_coord,y_top)

def round_to(x, base=5):
    return int(base * round(float(x)/base))


threshold = 70

video = sys.argv[1]
vidcap = cv2.VideoCapture(video)
success,image = vidcap.read()

# plt.ion()
# plt.figure()
show_frame(image)

# while success:
#     success,image = vidcap.read()
#     show_image(image)

pool_time = 3
pool_size = pool_time*fps*3
right_edge_pool = np.zeros(pool_size,dtype='int')
left_edge_pool = np.zeros(pool_size,dtype='int')
x_top = np.shape(image)[1]

time = 0.
while time < 5*1000.:

    _,image = vidcap.read()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    threshold_image = cv2.adaptiveThreshold(image, np.iinfo(image.dtype).max,\
     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,251,2)
    contour_list, _ = cv2.findContours(threshold_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour_list = order_by_area(contour_list)
    contour_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image,contour_list[1:4],-1,(0,0,255),2)

    for i in range(3):
        edgepoint = get_rightmost_point(contour_list[i+1])
        print(edgepoint)
        right_edge_pool[i] = edgepoint[0]
        # p1,p2 = get_vertical_border_endpoints(edgepoint[0],image)
        # print(p1,p2)
        # cv2.line(contour_image, p1,p2, (255,0,0), 2)

        edgepoint = get_leftmost_point(contour_list[i+1])
        left_edge_pool[i] = edgepoint[0]
        print(edgepoint)
        # p1,p2 = get_vertical_border_endpoints(edgepoint[0],image)
        # print(p1,p2)
        # cv2.line(contour_image, p1,p2, (0,255,0), 2)

    #calculated running mode
    if time>2000.:
        # print(right_edge_pool)
        right_edge_mode = scipy.stats.mode(
            right_edge_pool[right_edge_pool<x_top-5])[0][0]
        # print('rem: '+str(right_edge_mode))
        p1,p2 = get_vertical_border_endpoints(right_edge_mode,image)
        cv2.line(contour_image, p1,p2, (0,0,255), 2)
        left_edge_mode = scipy.stats.mode(
            left_edge_pool[left_edge_pool>right_edge_mode])[0][0]
        p1,p2 = get_vertical_border_endpoints(left_edge_mode,image)
        cv2.line(contour_image, p1,p2, (0,0,255), 2)

    #shift and drop pool edges
    right_edge_pool = np.concatenate((np.zeros(3, dtype = 'int'),right_edge_pool[0:pool_size-3]))
    left_edge_pool = np.concatenate((np.zeros(3, dtype = 'int'),left_edge_pool[0:pool_size-3]))
    show_frame(contour_image)
    time+= 1000./fps
    print time/1000.

#Now create a new image cropped by right_edge_mode and left_edge_mode

cropped_image = image[:,right_edge_mode:left_edge_mode]
show_image(cropped_image)
