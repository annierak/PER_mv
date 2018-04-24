import cv2

def show_image(img,size=800):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', size, size)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


vidcap = cv2.VideoCapture('NPF_2.avi')
_,frame = vidcap.read()
show_image(frame)
