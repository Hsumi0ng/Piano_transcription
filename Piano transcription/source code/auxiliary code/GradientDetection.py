import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
plt.rcParams['image.cmap'] = 'gray'

def DLine(img,h):
    line=img[h,:]
    dx=np.gradient(line)
    xlable=np.arange(0,len(dx))
    dx=h-dx
    return xlable,dx,line

#imagepath=r'D:\cvclass\assignment1\pythonProject\venv\imgs\piano1.png'
imagepath=r"C:\Users\PC\Desktop\grayscale1.png"
img=cv.imread(imagepath)
img1=img.copy()
gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
xlable,dx,line=DLine(gray,h=600)
plt.figure()
plt.imshow(gray)
plt.plot(xlable,line)
plt.show()


