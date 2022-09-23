import numpy as np
import cv2
from Library import *
import time as t
from get_keyboard import *
import random

def show(name,img):
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,1920,1080)
    cv2.imshow(name,img)

def DLine(img,h):
    line=img[h,:]
    dx=np.gradient(line)
    xlable=np.arange(0,len(dx))
    return xlable,line


def number_of_certain_probability(sequence, probability):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


value_list = [0, 1]
probability = [0.3125, 0.6875]
vidpath="vid.mp4"#vid or blackeys
#reg分帧
n = 5
reg = np.ones((n, 1))*(-1)
reg2 = np.ones((n, 1))*(-1)

cap = cv2.VideoCapture(vidpath)
ret, frame = cap.read()
FPS = int(cap.get(5))#1s多少帧

cap.set(cv2.CAP_PROP_POS_FRAMES,114)#114 or 1
_,background=cap.read()
background1 = cv2.resize(background, (int(background.shape[1]/2),int(background.shape[0]/2)))
cv_show(background1,"1")
minr, minc, maxr, maxc = get_keyboard_step1(background1, 170)
keyboard1, M1 = perspective_transform(minr, minc, maxr, maxc, background1)
keyboard2, M2 = get_keyboard_step2(keyboard1, 50)
xlable1, xlable2, dx1, dx2 = blackkey_gradient(keyboard2, 55, 45)
boundaryright, boundaryleft, boundarycentral1, boundarywhiteleft, boundarywhiteright,_ = divide_keyboard(keyboard2, xlable1, xlable2, dx1, dx2)
bg = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
cnt=0
DetectLineHeightW=560
DetectThresholdW=50
DetectLineHeightB=545
DetectThresholdB=97
fps=[]
time = []
dnotes = []
right = 0
while(cap.isOpened()):
    # print("time:", cnt)
    #time_start=t.time()
    cnt=cnt+1
    ret, frame = cap.read()
    if frame is None: break
    after = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #show("before",gs)
    afterw =cv2.subtract(bg,after) #black detection in grayscale
    afterb = cv2.subtract(after, bg)
    #show("bg",bg)
    #show("after",gs)
    '''
    if cnt==200:
        cv2.imwrite(path,img) #提取第几帧图像 可以用GradientDetection作测试 测出强度阈值和检测高度
    '''
    dnotestime = []
    dnotestime2 = []
    _,line=DLine(afterw,h=DetectLineHeightW)
    presslocation = []

    for i,x in enumerate(line):
        if x<DetectThresholdW: continue
        cv2.line(frame,(i,DetectLineHeightW),(i,DetectLineHeightW-x),(0,255,0),2)# 检测线强度  高度h=600
        presslocation.append(i)
        state = 0
        for j in range(len(presslocation)):
            if abs(i - presslocation[j]) < 20:
                if i != presslocation[j]:
                    state = 1
                    continue
        if state == 1:
            continue
        point = np.zeros((3, 1))
        point[0] = i/2
        point[1] = DetectLineHeightW/2
        point[2] = 1
        x1, _, _ = np.dot(np.dot(M2, M1), point)
        state = 0
        regreg = -1
        for j in range(52):
            if (x1 > boundarywhiteleft[j] ) and (x1 <= boundarywhiteright[j] ):
                regreg = j
        for j in range(n):
            if regreg == int(reg[j]):
                state = 1
        if state == 1: continue
        reg[1:n] = reg[0:n - 1]
        reg[0] = regreg
        # result = number_of_certain_probability(value_list, probability)        # print(regreg)
        # if result == 0:
        #     reg[1:n] = reg[0:n - 1]
        #     reg[0] = -1
        dnotestime.append(x1)
    _, line = DLine(afterb, h=DetectLineHeightB)
    for i, x in enumerate(line):
        if x < DetectThresholdB: continue  # 强度阈值
        # cv2.putText(frame, "max:"+str(max), (200, 300), cv2.FONT_HERSHEY_PLAIN, 10.0, (0, 0, 255), 2)# 在视频里给出强度
        cv2.line(frame, (i, DetectLineHeightB), (i, DetectLineHeightB - x), (0, 0, 255), 2)  # 检测线强度  高度h=600
        presslocation.append(i)
        state = 0
        for j in range(len(presslocation)):
            if abs(i - presslocation[j]) < 20:
                if i != presslocation[j]:
                    state = 1
                    continue
        if state == 1:
            continue
        point = np.zeros((3, 1))
        point[0] = i / 2
        point[1] = DetectLineHeightB / 2
        point[2] = 1
        x1, _, _ = np.dot(np.dot(M2, M1), point)
        state = 0
        regreg = -1
        for j in range(36):
            if (x1 > boundarycentral1[j] - 20) and (x1 <= boundarycentral1[j] +20):
                regreg = j
        for j in range(n):
            if regreg == int(reg[j]):
                state = 1
        if state == 1: continue
        reg2[1:n] = reg2[0:n - 1]
        reg[0] = regreg
        # print(regreg)
        dnotestime2.append(x1)
    # after = cv2.erode(after,(7,7),iterations = 1)
    if len(dnotestime) + len(dnotestime2) == 1:
        if len(dnotestime) != 0:
            for i in range(52):
                if (dnotestime[0] > boundarywhiteleft[i]) and (dnotestime[0] <= boundarywhiteright[i]):
                    dnotes.append(i)
                    #print(i)
        if len(dnotestime2) != 0:
            for i in range(36):
                if (dnotestime2[0] > boundarycentral1[i] - 20) and (dnotestime2[0] <= boundarycentral1[i] +20):
                    dnotes.append(i+52)
                    #print(i+52)
        time.append(int(cnt/FPS*1000)/1000)
    elif len(dnotestime) + len(dnotestime2) == 0:
        assert 1 == 1
    else:
        dnt = []
        if len(dnotestime) != 0:
            for j in range(len(dnotestime)):
                for i in range(52):
                    if (dnotestime[j] > boundarywhiteleft[i]) and (dnotestime[j] <= boundarywhiteright[i]):
                        dnt.append(i)
        if len(dnotestime2) != 0:
            for j in range(len(dnotestime2)):
                for i in range(36):
                    if (dnotestime2[0] > boundarycentral1[i] - 20) and (dnotestime2[0] <= boundarycentral1[i] + 20):
                        assert 1==1#dnt.append(i + 52)
        #print(dnt)
        time.append(int(cnt/FPS*1000)/1000)
        dnotes.append(dnt)

    cv2.namedWindow("image2",0)
    cv2.resizeWindow("image2", 1920,1080)

    cv2.imshow("image2",frame)
    #time_end=t.time()
    #fps.append(1/(time_end-time_start))
    #if cnt==5270:print(sum(fps)/len(fps))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('r'):
        right = right + 1
        #print(right)
cap.release()
cv2.destroyAllWindows()
#print(time, dnotes)
#print(right,len(dnotes),FPS)
