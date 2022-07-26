import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyzbar.pyzbar as pyzbar

width, height = 0, 0


def decode(img):
    # if height < width, then rotate the image
    img = img if img.shape[0] > img.shape[1] else np.rot90(img)

    global height, width
    width, height = img.shape[1], img.shape[0]

    roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roi_map = cv2.convertScaleAbs(roi_gray, alpha=1, beta=0)  # 將灰階轉換成黑白圖，增強對比度

    _, roi_thresh = cv2.threshold(
        roi_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法閾值化

    # 一些可能用的功能
    # roi_thresh = cv2.bitwise_not(roi_thresh)  # 反轉圖像

    roi_thresh = cv2.inpaint(roi_thresh, np.zeros(
        roi_thresh.shape, dtype=np.uint8), 3, cv2.INPAINT_TELEA)  # 填充黑色區域

    roi_thresh = fill_color(roi_thresh, 255)
    # Border
    borders = 250
    canvas = np.zeros((height+2*borders, width+2*borders), dtype=np.uint8)
    canvas[:] = 255

    canvas[borders:-borders, borders:-borders] = roi_thresh

    # cv2.imshow('canvas', canvas)
    # cv2.waitKey(0)

    return process1(canvas)


def process1(img):
    img_cp = img.copy()
    roi_blur = cv2.GaussianBlur(img_cp, (5, 5), 0)
    roi_edge = cv2.Canny(roi_blur, 50, 100, apertureSize=3)

    contours, _ = cv2.findContours(
        roi_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找輪廓

    # 依照凸包面積大小來排序
    contours = sorted(contours, key=lambda c: cv2.contourArea(
        cv2.convexHull(c)), reverse=True)

    contours = [contours[0]]  # 只留下最大的輪廓

    # # show image
    # cp = roi_edge.copy()
    # convex = [cv2.convexHull(contours) for contours in contours]  # 找輪廓的凸包
    # cv2.polylines(cp, convex, True, 255, 3)
    # cv2.namedWindow("convex", cv2.WINDOW_NORMAL)
    # cv2.imshow("convex", cp)

    # 畫出凸包
    convex = draw_approx_hull_polygon(img_cp, contours)  # 繪製凸包
    # 取得角點
    pts = getcorner(convex)

    # pt0--------pt1
    # |           |
    # pt3--------pt2
    h, w = pts[3][1]-pts[0][1], pts[2][0]-pts[3][0]

    img = Perspective(img, pts, w, h)

    return process2(img)


def process2(img):
    # 上一步已經取得了QR code的4個角點，並根據角點的位置
    # 把圖片不需要的部分裁切掉了

    # 去除黑框

    dst = fill_color(img, 255)

    cv2.inpaint(dst, np.ones(dst.shape, dtype=np.uint8),
                3, cv2.INPAINT_TELEA, dst)
    # cv2.imshow('dst', dst)
    dst = np.where(dst < 120, 0, 255).astype(np.uint8)
    # 我們現在要找的東西是這樣的:

    # w w w w w w w w #   <- start #1
    # ............... #
    #                 #
    #     上半部
    #                 #
    # ............... #
    # w w w w w w w w #   <- end #1
    # w w w w w w w w #   <- start #2
    # *************** #
    #
    #     下半部
    #
    # *************** #
    # w w w w w w w w #   <- end #2

    # wt_row = np.where((dst == np.full(w, 255, dtype=np.uint8)).all(axis=1))[0]
    # if len(wt_row) < 3:
    #     return None

    # 先找到上半部的起始和結束點
    h, w = np.int0(dst.shape)

    res = np.where((dst == np.full(w, 255, dtype=np.uint8)).all(axis=1))[0]
    y1 = res[np.where((res < h/10))][-1]
    y2 = res[np.where((res > h/3) & (res < h*2/3))][0]  # <- start #2
    y3 = res[np.where((res > h/3) & (res < h*2/3))][-1]  # <- end #1, start #2
    y4 = res[np.where((res > h*9/10))][0]   # <- end #2

    half_1 = dst[y1:y2, :]
    half_2 = dst[y3:y4, :]

    half_2 = np.flip(half_2, axis=0)  # 翻轉

    # # resize
    # ratio = half_1.shape[0] / half_2.shape[0]
    # h2, w2 = half_1.shape[0], int(half_2.shape[1]*ratio)
    # half_2 = cv2.resize(half_2, (w2, h2), interpolation=cv2.INTER_CUBIC)

    half_2 = cv2.resize(half_2, (half_2.shape[1], half_1.shape[0]))

    half_1 = fill_color(half_1, 255)
    half_2 = fill_color(half_2, 255)

    return process3(half_1, half_2)


def process3(half_1, half_2):

    half_1 = np.where(half_1 < 120, 0, 255).astype("uint8")
    half_2 = np.where(half_2 < 120, 0, 255).astype("uint8")

    w1, w2 = half_1.shape[1], half_2.shape[1]
    x1, x2 = w1-1, 0

    # 這個部分已經驗證過了，如果用一次多行的方式計算白色比例，最終會有微妙的誤差，
    # 這個微妙的誤差會導致最後decode失敗(真的是龜公)
    for x in range(w1-1, 0, -1):

        if (half_1[:, x] == 255).sum() / (half_1.shape[0]) < 0.7:
            x1 = x
            break
    for x in range(0, w2):
        if (half_2[:, x] == 255).sum() / (half_2.shape[0]) < 0.7:
            x2 = x
            break

    flag = 1
    for x in range(x2, w2//2):
        col = half_2[:, x]
        cnt = 0
        for c in range(1, len(col)):
            if col[c-1] < 120 and col[c] > 120:
                cnt += 1
        if cnt < 9 and flag == 0:
            x2 = x
            break
        if flag == 1 and cnt > 9:
            flag -= 1
        # print(cnt)

    # cv2.namedWindow("half_1", cv2.WINDOW_NORMAL)
    # cv2.imshow('half_1', half_1[:, :x1])
    # cv2.namedWindow("half_2", cv2.WINDOW_NORMAL)
    # cv2.imshow('half_2', half_2[:, x2:])

    qr_code = np.hstack((half_1[:, :x1], half_2[:, x2:]))

    return show(qr_code)


def show(qr_code):
    borders = round(qr_code.shape[0]/21*4)
    canvas = np.zeros(
        (qr_code.shape[0]+2*borders, qr_code.shape[1]+2*borders), dtype=np.uint8)
    canvas[:, :] = 255
    canvas[borders:qr_code.shape[0]+borders,
           borders:qr_code.shape[1]+borders] = qr_code
    # cv2.imshow('qr_code', canvas)
    # cv2.imwrite('qr_code-2.png', canvas)
    # cv2.waitKey(0)

    text = pyzbar.decode(canvas)
    if text == []:
        blur = cv2.GaussianBlur(canvas, (3, 3), 0)
        # closing = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.zeros((3, 3), np.uint8))
        text = pyzbar.decode(blur)
        cv2.imshow('qr_code', blur)

    if text == []:
        return None
    else:
        return text[0].data.decode('utf-8')


def getcorner(canny):
    global width, height
    # shi-tomasi corner detection
    corners = cv2.goodFeaturesToTrack(
        canny, 4, 0.01, width//8, blockSize=10, useHarrisDetector=False, k=0.04)

    # # 另一種方法：亞像素角偵測
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    # cv2.cornerSubPix(canny, np.float32(corners), (5, 5), (-1, -1), criteria)

    corners = np.int0(corners)
    pts = []
    for c in corners:
        x, y = c.ravel()
        pts.append([x, y])

    if len(pts) != 4:
        return None

    def sort(pt):
        pt = sorted(pt, key=lambda x: x[1])
        if pt[0][0] > pt[1][0]:
            pt[0], pt[1] = pt[1], pt[0]
        if pt[3][0] > pt[2][0]:
            pt[2], pt[3] = pt[3], pt[2]
        return pt

    pts = sort(pts)
    # # 畫出角點
    # for x, y in pts:
    #     cv2.circle(canny, (x, y), 3, (0, 0, 255), -1)
    # cv2.imshow('corners', canny)

    return pts


def draw_approx_hull_polygon(img, contours):  # 畫出多邊形
    # 取出多邊形凸包
    hulls = [cv2.convexHull(cnt) for cnt in contours]  # 找輪廓的凸包

    # show image
    img_cp = np.zeros(img.shape, dtype=np.uint8)
    cv2.polylines(img_cp, hulls, isClosed=True, color=255, thickness=1)
    # cv2.imshow('convex hull', img_cp)

    return img_cp


def fill_color(img, color=255):
    h, w = img.shape[:2]
    img = np.asarray(img)
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    seedPoint = [(0, 0), (0, h-1), (w-1, h-1), (w-1, 0)]
    for sp in seedPoint:
        cv2.floodFill(img, None, sp, (color, color, color))
    # floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None)
    return img


def Perspective(image, pts1, w, h):
    w, h = int(w), int(h)
    if w > h:
        w, h = h, w
    # src: 源點
    pts1 = np.float32(pts1)
    # dst: 目標點（必須以與src點相同的順序列出！）
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # 轉換矩陣
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 套用矩陣
    dst = cv2.warpPerspective(image, M, (w, h))
    return dst
