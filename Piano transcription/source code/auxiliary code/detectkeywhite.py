import numpy as np
import cv2
from Library import *
import time as t

def show(name,img):
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,1920,1080)
    cv2.imshow(name,img)

def DLine(img,h):
    line=img[h,:]
    dx=np.gradient(line)
    xlable=np.arange(0,len(dx))
    return xlable,line

vidpath=r"D:\cvclass\assignment1\pythonProject\venv\vids\vid.mp4"#vid or blackeys
cap = cv2.VideoCapture(vidpath)
ret, frame = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES,114)#114 or 1
_,background=cap.read()
bg = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
cnt=0
DetectLineHeightW=560
DetectThresholdW=50
fps=[]
while(cap.isOpened()):
    #time_start=t.time()
    cnt=cnt+1
    ret, frame = cap.read()
    if frame is None: break
    gs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #show("before",gs)
    gs=cv2.subtract(bg,gs) #black detection in grayscale
    #show("bg",bg)
    #show("after",gs)
    '''
    if cnt==200:
        cv2.imwrite(path,img) #提取第几帧图像 可以用GradientDetection作测试 测出强度阈值和检测高度
    '''
    _,line=DLine(gs,h=DetectLineHeightW)
    #r=cr_otsu(frame,background)
    for i,x in enumerate(line):
     #   if len(r)==0:continue
      #  if i<r[0]-50:continue
       # if i>r[-1]+50:continue
        if x<DetectThresholdW: continue
        cv2.line(frame,(i,DetectLineHeightW),(i,DetectLineHeightW-x),(0,255,0),2)# 检测线强度  高度h=600
    cv2.namedWindow("image2",0)
    cv2.resizeWindow("image2", 1920,1080)
    cv2.imshow("image2",frame)
    #time_end=t.time()
    #fps.append(1/(time_end-time_start))
    #if cnt==5270:print(sum(fps)/len(fps)) 测试fps代码
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()