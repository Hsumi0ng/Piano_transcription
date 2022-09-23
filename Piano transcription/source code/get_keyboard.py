from cv_func import *
import numpy as np
from skimage import measure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pylab
import cv2


# 输入原彩图;输入二值化阈值以调节连通矩形面积使键盘矩形完整且面积最大
# 输出矩形四个点信息（矩形x，y方向最小值与最大值，注意这个矩形的两条垂直边分别与x，y轴平行）
def get_keyboard_step1(original_img, threshold):
    # 灰度图-二值化
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, binaryimg = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel_shape = 3
    kernel = np.ones((kernel_shape, kernel_shape), np.uint8)
    erode = cv2.erode(binaryimg, kernel, iterations=2)
    dilation = cv2.dilate(erode, kernel, iterations=6)
    # 连通块
    labeled_img, num = measure.label(dilation, connectivity=1, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(0, num):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    mcr = (labeled_img == max_label)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(mcr)
    count = 0
    area = np.zeros([num, 1])
    for region in measure.regionprops(labeled_img):
        area[count] = region.area
        count = count + 1
    maxarea = np.max(area)
    minr, minc, maxr, maxc = 0, 0, 0, 0
    for region in measure.regionprops(labeled_img):
        if region.area == maxarea:
            minr, minc, maxr, maxc = region.bbox

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    #pylab.show()
    return minr, minc, maxr, maxc


# 第一次图像透射变换,从左侧拍和右侧拍在这里pts1设置不同
# 输入矩形四个点信息（minr, minc, maxr, maxc），以及原彩图
# 输出变换后的图以及变换矩阵
def perspective_transform(minr, minc, maxr, maxc, original_img):
    k = -60
    pts1 = np.float32([[minc+k, minr], [maxc, minr], [minc+k, maxr], [maxc, maxr]])
    w = maxc - minc-k
    h = maxr - minr
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(original_img, M, (w, h))
    #cv_show(dst, "dst")
    return dst, M


# 通过边缘检测，提取键盘四个边界点，并作透射变换
# 输入step1矩形彩图，输入二值化阈值
# 输出w=1248,h=194的键盘与变换矩阵
def get_keyboard_step2(src, threshold):
    keyboard1 = src.copy()
    gray = cv2.cvtColor(keyboard1, cv2.COLOR_BGR2GRAY)
    _, binaryimg = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    h, w = binaryimg.shape
    kernel_shape = 3
    kernel = np.ones((kernel_shape, kernel_shape), np.uint8)
    erode = cv2.erode(binaryimg, kernel, iterations=2)
    dilation = cv2.dilate(erode, kernel, iterations=2)
    edges = cv2.Canny(dilation, 50, 150, apertureSize=3)
    #cv_show(edges,"1")
    edge = np.where(edges == 255)
    edgelocation = np.zeros([2, edge[0].shape[0]])
    edgelocation[0] = edge[1]
    edgelocation[1] = edge[0]
    edgelocation = edgelocation.T
    x1 = np.argmin(edgelocation[:, 1] + edgelocation[:, 0] * np.divide(h, w))
    x2 = np.argmin(edgelocation[:, 1] - edgelocation[:, 0] * np.divide(h, w))
    x3 = np.argmax(edgelocation[:, 1] - edgelocation[:, 0] * np.divide(h, w))
    x4 = np.argmax(edgelocation[:, 1] + edgelocation[:, 0] * np.divide(h, w))
    pts1 = np.float32([[edgelocation[x1, 0], edgelocation[x1, 1]],
                       [edgelocation[x2, 0], edgelocation[x2, 1]],
                       [edgelocation[x3, 0], edgelocation[x3, 1]],
                       [edgelocation[x4, 0], edgelocation[x4, 1]]])
    w = 1248
    h = 194
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(keyboard1, M, (w, h))
    #cv_show(dst, "copyimg")
    return dst, M


# 梯度求黑键左右边缘
# 输入两个阈值，输出标签及梯度
def blackkey_gradient(src, threshold1, threshold2):
    keyboard2 = src.copy()

    # 梯度求黑键右边缘
    gray = cv2.cvtColor(keyboard2, cv2.COLOR_RGB2GRAY)
    ret, gs = cv2.threshold(gray, threshold1, 255, cv2.THRESH_BINARY)
    H, W = gs.shape
    xlable1, dx1 = DLine(gs, h=int(H / 2))
    # plt.figure()
    # plt.imshow(gs)
    # plt.plot(xlable1, dx1)
    # plt.show()

    # 梯度求黑键左边缘
    gray = cv2.cvtColor(keyboard2, cv2.COLOR_RGB2GRAY)
    ret, gs = cv2.threshold(gray, threshold2, 255, cv2.THRESH_BINARY_INV)
    H, W = gs.shape
    xlable2, dx2 = DLine(gs, h=int(H / 2))
    # plt.figure()
    # plt.imshow(gs)
    # plt.plot(xlable2, dx2)
    # plt.show()
    return xlable1, xlable2, dx1, dx2


# 梯度分块
def divide_keyboard(src, xlable1, xlable2, dx1, dx2):
    keyboard2 = src.copy()
    w = 1248
    h = 194
    key_result = np.ones((h,w))
    key_result = key_result*255

    boundaryleft = []
    boundaryright = []
    reg = -100
    for i in xlable1:
        if dx1[i] < 97 - 10 and (i - reg > 20):
            boundaryleft.append(i)
            reg = i
    reg = -100
    for i in xlable2:
        if dx2[i] < 97 - 10 and (i - reg > 20):
            boundaryright.append(i)
            reg = i
    print(len(boundaryright), len(boundaryleft))
    # cv_show(keyboard2,"1")
    assert len(boundaryleft) == 36
    assert len(boundaryright) == 36

    boundarycentralindex = [0, 2, 3, 5, 6, 7, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 23, 24,
                            26, 27, 28, 30, 31, 33, 34, 35, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49]
    boundarycentral1 = []
    boundarywhiteleft = []
    boundarywhiteright = []
    boundarywhiteleft.append(0)
    count = 0
    # boundaryright的index
    for i in range(52 - 1):
        if i in boundarycentralindex:
            boundarycentral = (boundaryright[count] + boundaryleft[count]) / 2
            boundarycentral1.append(boundarycentral)
            boundarywhiteleft.append(boundarycentral)
            boundarywhiteright.append(boundarycentral)
            count = count + 1
            ptStart = (int(boundarycentral), int(h / 3 * 2))
            ptEnd = (int(boundarycentral), h)
            point_color = (0, 0, 255)  # BGR
            thickness = 1
            lineType = 4
            cv2.line(keyboard2, ptStart, ptEnd, point_color, thickness, lineType)
        elif count != 36:
            boundarycentral = (boundaryright[count - 1] + boundaryleft[count]) / 2
            boundarywhiteleft.append(boundarycentral)
            boundarywhiteright.append(boundarycentral)
            ptStart = (int(boundarycentral), 0)
            ptEnd = (int(boundarycentral), h)
            point_color = (0, 0, 255)  # BGR
            thickness = 1
            lineType = 4
            cv2.line(keyboard2, ptStart, ptEnd, point_color, thickness, lineType)
        else:
            boundarycentral = boundarycentral1[35] * 2 - boundarycentral1[34]
            boundarywhiteleft.append(boundarycentral)
            boundarywhiteright.append(boundarycentral)
            ptStart = (int(boundarycentral), 0)
            ptEnd = (int(boundarycentral), h)


            point_color = (0, 0, 255)  # BGR
            thickness = 1
            lineType = 4
            cv2.line(keyboard2, ptStart, ptEnd, point_color, thickness, lineType)
    boundarywhiteright.append(w)
    for i in range(36):
        center = (boundaryright[i] + boundaryleft[i]) / 2
        blackw = (boundaryright[i] - boundaryleft[i])
        # 10*2
        dstrgb1 = cv2.rectangle(keyboard2, (int(center - (blackw / 2)), 0),
                                (int(center + (blackw / 2)), int(h / 3 * 2)), (0, 255, 0), 2)
        key_result[int(h / 3 * 1):int(h / 3 * 2), int(center + (blackw / 2)):int(center - (blackw / 2))] = 0
    cv_show(dstrgb1, "keyboard")
    assert len(boundarywhiteright) == 52
    assert len(boundarywhiteleft) == 52
    return boundaryright, boundaryleft, boundarycentral1, boundarywhiteleft, boundarywhiteright,key_result

