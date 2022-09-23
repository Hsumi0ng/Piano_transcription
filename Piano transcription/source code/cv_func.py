import cv2
import numpy as np


def cv_show(img, name, time=0):
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()

def DLine(img,h):
    line=img[h,:]
    dx=np.gradient(line)
    xlable=np.arange(0,len(dx))
    dx=h-dx
    return xlable,dx


