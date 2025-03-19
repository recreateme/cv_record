import cv2
import numpy as np
import subprocess

# 鼠标回调设置
drawing = False  # 鼠标按下时为True
ix, iy = -1, -1  # 初始坐标
rect_params = []  # 存储矩形参数


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect_params

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_params.extend([ix, iy, x, y])
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)



img = cv2.imread('star.png')
img_copy = img.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        break

# 保存矩形位置到txt文件
with open('analyrect.txt', 'w') as f:
    str_list = list(map(str, rect_params))
    rs = ','.join(str_list)
    f.write(f'{rs}')

cv2.destroyAllWindows()
subprocess.run('python star.py')
