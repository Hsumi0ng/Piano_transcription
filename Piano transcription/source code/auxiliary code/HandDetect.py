import numpy as np
import cv2
def show(name,img):
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,1920,1080)
    cv2.imshow(name,img)

def cr_otsu(img,cnt,bg):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.namedWindow("image raw", cv2.WINDOW_NORMAL)
    xor=cv2.subtract(img,bg)
    #dst = cv2.bitwise_and(xor, xor, mask=skin)
    dst=cv2.cvtColor(xor, cv2.COLOR_RGB2GRAY)
    dst=dst[749,:]
    dst[dst>=80]=255
    dst[dst<80]=0
    r=np.where(dst>0)[0]
    cv2.line(img,(0,749),(1919,749), (0,255,0),5)
    if len(r)==0:pass
    else:
        left=r[0]-50
        right=r[-1]+50
        cv2.line(img,(left,0),(r[0]-50,1080), (0,255,0),5)
        cv2.line(img,(r[-1]+50,0),(r[-1]+50,1080), (0,255,0),5)
        cv2.line(img,(int((left+right)/2)+10,0),(int((left+right)/2)+10,1080), (0,255,0),5)
        print(left,right)
    cv2.imshow("image raw", img)
    cv2.waitKey(0)
    return

vidpath=r"D:\cvclass\assignment1\pythonProject\venv\vids\vid.mp4"
cnt=0
cap = cv2.VideoCapture(vidpath)
cap.set(cv2.CAP_PROP_POS_FRAMES,114)
_,background=cap.read()
ret, frame = cap.read()
while(cap.isOpened()):
    ret, frame = cap.read()
    cr_otsu(frame,cnt,background)
    cnt=cnt+1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

