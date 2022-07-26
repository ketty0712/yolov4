import argparse
import os
# from multiprocessing import cpu_count
from queue import Queue

import cv2

import darknet.darknet_images as dn
import opencv.recover as qrcode

'''
如果只是單純運行測試，可以把 def image(f="...") 裡面的參數改成要測試圖片路徑
把 32 行註解
把 33 行反註解
'''


def image(f="./darknet/data/QRCode/image25.jpg"):

    bookmarks = set()
    args = dn.parser()
    # print(args)
    args.input = os.path.join(os.path.dirname(__file__), f)
    img_for_decode = dn.main(args)

    while not img_for_decode.empty():
        img = img_for_decode.get()
        try:
            bookmarks.add(qrcode.decode(img))
            # print(bookmarks)
            # cv2.waitKey()
        except:
            continue
    return bookmarks
    # output_file(bookmarks)


def output_file(bookmarks, filename="output.txt"):
    with open(filename, "w") as f:
        for bookmark in bookmarks:
            if bookmark != None:
                f.writelines(bookmark)


if __name__ == '__main__':
    # image('D:/Yolo_v4_cp/image_uploads/1-4-A-1-1-1/1-4-A-1-1-1-0.jpg')
    image()