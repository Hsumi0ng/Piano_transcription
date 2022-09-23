import numpy as np
import cv2

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
frame = cv2.transpose(frame)
cap.set(cv2.CAP_PROP_POS_FRAMES,114)
_,background=cap.read()
bg = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
cnt=0
DetectLineHeightB=545
DetectThresholdB=97
while(cap.isOpened()):
    print("time:",cnt)
    cnt=cnt+1
    ret, frame = cap.read()
    after = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    show("before",after)
    after=cv2.subtract(after,bg) #black detection in grayscale
    show("bg",bg)
    show("after",after)
    _,line=DLine(after,h=DetectLineHeightB)
    max=np.max(line)
    for i,x in enumerate(line):
        if x<DetectThresholdB:continue #强度阈值
        #cv2.putText(frame, "max:"+str(max), (200, 300), cv2.FONT_HERSHEY_PLAIN, 10.0, (0, 0, 255), 2)# 在视频里给出强度
        cv2.line(frame,(i,DetectLineHeightB),(i,DetectLineHeightB-x),(0,255,0),2)# 检测线强度  高度h=600
    #after = cv2.erode(after,(7,7),iterations = 1)
    cv2.namedWindow("image2",0)
    cv2.resizeWindow("image2", 1920,1080)
    cv2.imshow("image2",frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()