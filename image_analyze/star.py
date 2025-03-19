import os

import cv2
import numpy as np
from scipy import optimize
from scipy import signal

import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


def find_star_field_center(image):
    image = np.array(image)
    rows, cols = image.shape
    max_pos = [rows // 2, cols // 2]
    k = 15
    # 遍历所有可能的位置
    max_wsum = 0
    for i in range(0, rows - k + 1, 3):
        for j in range(0, cols - k + 1, 3):
            # 计算子区域的和
            region_sum = np.sum(image[i:i + k, j:j + k])

            # 更新最大值
            if region_sum > max_wsum:
                max_sum = region_sum
                max_pos = (i, j)
    return max_pos[0] + k // 2, max_pos[1] + k // 2


class Point:
    def __init__(self, y, x):
        self.x = float(x)
        self.y = float(y)

    def __len__(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class Line:
    def __init__(self, point1, point2):
        self.A = point1
        self.B = point2

    def distance_to(self, pointC):
        # 计算点pointC到线的距离
        distance = np.abs(
            (pointC.y - self.A.y) * (self.B.x - self.A.x) - (pointC.x - self.A.x) * (self.B.y - self.A.y)) / np.sqrt(
            (self.B.x - self.A.x) ** 2 + (self.B.y - self.A.y) ** 2)
        return distance


def distance(p, lines):
    # 计算p点到所有lines距离的最大值
    return max(line.distance_to(Point(p[0], p[1])) for line in lines)


def process_mutipeaks(values, peak_ids):
    ids = []
    for i in range(len(peak_ids)):
        start = end = peak_ids[i]
        v1 = values[start]
        while True:
            end += 1
            if values[end] >= v1:
                continue
            break
        ids.append((start + end - 1) // 2)
    return ids


def check_reverse(img, percentiles=[4, 50, 96]):
    img = np.array(img, np.uint8)
    p_low = np.percentile(img, percentiles[0])
    p_mid = np.percentile(img, percentiles[1])
    p_high = np.percentile(img, percentiles[2])
    mid_to_low = abs(p_mid - p_low)
    mid_to_high = abs(p_mid - p_high)
    if mid_to_low > mid_to_high:
        img = 255 - img
    return img


def is_descend(values, k=8):
    # values应该是一维数组
    assert len(values) > k, f"传入的values长度小于{k}"
    values = np.array(values, dtype=np.float64)
    descend = np.where(values[:k] > values[1:k + 1], 1, 0)
    if descend.sum() >= 0.5 * k:
        return True
    return False


def satisfied(array, start, k=4):
    # median = np.median(array)
    array = np.squeeze(array)
    p80 = np.percentile(array, 80)
    if array[start:start + k].sum() <= k * (p80 + 1):
        return True
    return False


def preprocess(img):
    # 原始图像
    # cv2.imshow("raw", img)
    # cv2.waitKey(0)
    # 确认反转图像
    reversed = check_reverse(img)
    # cv2.imshow("reversed", reversed)
    # cv2.waitKey(0)

    # 滤波图像
    blur = cv2.medianBlur(reversed, 3)
    # cv2.imwrite("blur.png", blur)
    # cv2.imshow("blur", blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 寻找起始位置
    pts = []
    # 1-左上角
    row_off = 0
    col_off = 0
    row, col = blur.shape
    while True:
        # current = blur[row_off, col_off]
        # q1_row, median_row, q2_row = np.percentile(blur[row_off, :], [5, 50, 75])
        # q1_col, median_col, q2_col = np.percentile(blur[:, col_off], [5, 50, 75])
        # 起始位置要满足要求，不会是一个大数
        if not (satisfied(blur[row_off, :], col_off, k=3) and satisfied(blur[:, col_off], row_off, k=5)):
            # if current >= q2_row or current >= q2_col:
            row_off += 7
            continue
        flag1 = is_descend(blur[row_off, col_off:])
        flag2 = is_descend(blur[row_off:, col_off])

        descend = flag1 or flag2
        if descend:
            row_off += 7
            continue
        pts.append((row_off, col_off))
        break
    # 2-右上角
    row_off = pts[0][0]
    col_off = blur.shape[1] - 1
    while True:
        current = blur[row_off, col_off]
        q1_row, median_row, q2_row = np.percentile(blur[row_off, :], [5, 50, 75])
        q1_col, median_col, q2_col = np.percentile(blur[:, col_off], [5, 50, 75])
        # if current >= q2_row or current >= q2_col:
        if not (satisfied(blur[row_off, :][::-1], col - col_off - 1, k=3) and satisfied(blur[:, col_off], row_off,
                                                                                        k=5)):
            col_off -= 7
            continue
        descend = is_descend(blur[row_off, :col_off][::-1]) or is_descend(blur[row_off:, col_off])
        if descend:
            col_off -= 7
            continue
        pts.append((row_off, col_off))
        break
    # 3-左下角 && 右下角
    row_off = row - 1
    col_off1 = pts[0][1]
    col_off2 = pts[1][1]
    while True:
        # current1 = blur[row_off, col_off1]
        # current2 = blur[row_off, col_off2]
        # q1_row, median_row, q2_row = np.percentile(blur[row_off, :], [5, 50, 75])
        # q1_col1, median_col1, q2_col1 = np.percentile(blur[:, col_off1], [5, 50, 75])
        # q1_col2, median_col2, q2_col2 = np.percentile(blur[:, col_off2], [5, 50, 75])
        # if current1 >= q2_row or current1 >= q2_col1:
        if not (satisfied(blur[row_off, :], col_off1, k=3) and satisfied(blur[:, col_off1][::-1], row - row_off - 1,
                                                                         k=5)):
            row_off -= 7
            continue
        elif not (satisfied(blur[row_off, :][::-1], col - 1 - col_off2, k=3) and satisfied(blur[:, col_off2][::-1],
                                                                                           row - row_off - 1, k=5)):
            # elif current2 >= q2_row or current2 >= q2_col2:
            col_off2 -= 7
            continue
        descend1 = is_descend(blur[:row_off, col_off1][::-1]) or is_descend(blur[row_off, col_off1:])
        descend2 = is_descend(blur[:row_off, col_off2][::-1]) or is_descend(blur[row_off, :col_off2][::-1])
        if descend1 or descend2:
            row_off -= 7
            continue
        pts.append((row_off, col_off1))
        pts.append((row_off, col_off2))
        break

    return blur, pts


def main():
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    filename = 'star.png'
    crop_file = 'analyrect.txt'
    manual_txt = "star.txt"
    image = cv2.imread(filename, 0)
    if image is None:
        raise ValueError("Image not found")
    cv2.imshow("raw", image)
    cv2.waitKey(0)
    if os.path.exists(crop_file):
        crop_file = open(crop_file, "r")
        pos = crop_file.readline().strip().split(",")
        pos = [int(x) for x in pos]
        x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
        crop_file.close()
        image = image[y1:y2 + 1, x1:x2 + 1]
    cv2.imshow("crop", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not os.path.exists(manual_txt):
        blur, pts = preprocess(image)
        # print(f"shape: {blur.shape}, points: {pts}")
        cv2.imshow("blur1", blur)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

        start_point = Point(pts[0][0], pts[0][1])
        crop_img = blur[pts[0][0]:pts[2][0], pts[0][1]:pts[1][1]]
        cv2.imshow("crop_img", crop_img)
        cv2.waitKey(0)
        # center = find_star_field_center(crop_img)
        k = 0.15
        while True:
            values_ceil = np.array(crop_img[0, :])
            values_floor = np.array(crop_img[-1, :])
            values_left = np.array(crop_img[:, 0])
            values_right = np.array(crop_img[:, -1])

            k_max = 20
            row, col = crop_img.shape
            min_dis_row = row // 10
            min_dis_col = col // 10

            peaks_ceil, _ = signal.find_peaks(values_ceil, height=np.percentile(values_ceil, 80), distance=min_dis_col,
                                              prominence=k * (values_ceil.max() - values_ceil.min()))
            peaks_floor, _ = signal.find_peaks(values_floor, height=np.percentile(values_floor, 80),
                                               distance=min_dis_col,
                                               prominence=k * (values_floor.max() - values_floor.min()))
            peaks_left, _ = signal.find_peaks(values_left, height=np.percentile(values_left, 80), distance=min_dis_row,
                                              prominence=k * (values_left.max() - values_left.min()))
            peaks_right, _ = signal.find_peaks(values_right, height=np.percentile(values_right, 80),
                                               distance=min_dis_row,
                                               prominence=k * (values_right.max() - values_right.min()))

            if len(peaks_ceil) + len(peaks_floor) + len(peaks_left) + len(peaks_right) % 2 != 0:
                crop_img = crop_img[:-3, :]
            # 调整峰值位置
            peaks_ceil = process_mutipeaks(values_ceil, peaks_ceil)
            peaks_floor = process_mutipeaks(values_floor, peaks_floor)
            peaks_left = process_mutipeaks(values_left, peaks_left)
            peaks_right = process_mutipeaks(values_right, peaks_right)
            break

        # 可视化
        # figure, ax = plt.subplots(4, 1, figsize=(10, 10))
        # ax[0].plot(list(range(1, len(values_left) + 1)), values_left, "k", label='left')
        # ax[1].plot(list(range(1, len(values_right) + 1)), values_right, "b", label='right')
        # ax[2].plot(list(range(1, len(values_floor) + 1)), values_floor, "r", label='floor')
        # ax[3].plot(list(range(1, len(values_ceil) + 1)), values_ceil, '#775588', label='ceil')
        # # 在峰值位置标注
        # for peak in peaks_left:
        #     ax[0].axvline(x=peak, color='r', linestyle='--')
        # for peak in peaks_right:
        #     ax[1].axvline(x=peak, color='r', linestyle='--')
        # for peak in peaks_floor:
        #     ax[2].axvline(x=peak, color='r', linestyle='--')
        # for peak in peaks_ceil:
        #     ax[3].axvline(x=peak, color='r', linestyle='--')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        row, col = crop_img.shape
        pts_ceil = [Point(0, x) for x in peaks_ceil]
        pts_floor = [Point(row - 1, y) for y in peaks_floor]
        pts_left = [Point(x, 0) for x in peaks_left]
        pts_right = [Point(x, col - 1) for x in peaks_right]

        pts1 = []
        pts1.extend(pts_ceil)
        pts1.extend(pts_right)
        pts1.extend(pts_floor[::-1])
        pts1.extend(pts_left[::-1])
        num = len(pts1) // 2
        lines = [Line(pointA, pointB) for pointA, pointB in zip(pts1[:num], pts1[num:])]

        # 中心点
        res = optimize.minimize(
            distance,
            x0=[row // 2, col // 2],
            args=(lines,),
            method="Nelder-Mead",
            options={"fatol": 0.001},
        )

        radius = res.fun
        center = Point(res.x[0], res.x[1])

        center += start_point

        blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
        for line in lines:
            line.A += start_point
            line.B += start_point
            cv2.line(blur, (int(line.A.y), int(line.A.x)), (int(line.B.y), int(line.B.x)), (0, 0, 255), 1)
            cv2.circle(blur, (int(center.y), int(center.x)), 10, (0, 255, 0), 2)
        cv2.imshow("crop_img", blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        f = open('./star', 'w')
        f.write(f"{center.y},{center.x}\n")
        f.write(f"{radius}\n")
        # 角度
        degrees = []
        for line in lines:
            theta = np.degrees(np.arctan((line.B.x - line.A.x) / (line.B.y - line.A.y)))
            degrees.append((theta + 90) % 180)
        degrees = np.array(degrees)
        seq = np.argsort(degrees)
        for i in range(len(seq)):
            f.write(f"{lines[seq[i]].A.y},{lines[seq[i]].A.x},")
            f.write(f"{lines[seq[i]].B.y},{lines[seq[i]].B.x},")
            f.write(f"{degrees[seq[i]]}\n")
    else:
        img = cv2.imread(filename, 0)
        startPoint = [img.shape[0] // 2, img.shape[1] // 2]
        lines = []
        with open(manual_txt, "r") as f:
            for line in f.readlines():
                line = line.strip().split(',')
                pointA = Point(line[1], line[0])
                pointB = Point(line[3], line[2])
                lines.append(Line(pointA, pointB))

        res = optimize.minimize(
            distance,
            x0=startPoint,
            args=(lines,),
            method="Nelder-Mead",
            options={"fatol": 0.001},
        )

        radius = res.fun
        center = Point(res.x[0], res.x[1])
        f = open('./star', 'w')
        f.write(f"{center.x},{center.y}\n")
        f.write(f"{radius}\n")

        degrees = []
        for line in lines:
            # line.
            theta = (np.degrees(np.arctan((line.B.y - line.A.y) / (line.B.x - line.A.x))) + 90) % 180
            # degrees.append(theta)
            degrees.append((theta + 90) % 180)
        degrees = np.array(degrees)
        seq = np.argsort(degrees)
        for i in range(len(seq)):
            f.write(f"{lines[seq[i]].A.x},{lines[seq[i]].A.y},")
            f.write(f"{lines[seq[i]].B.x},{lines[seq[i]].B.y},")
            f.write(f"{degrees[seq[i]]}\n")
    # 可视化
    # iamge = cv2.imread('star2.png', 1)
    # for line in lines:
    #     cv2.line(iamge, (int(line.A.y), int(line.A.x)), (int(line.B.y), int(line.B.x)), (0, 255, 0), 1)
    # cv2.imshow("result", iamge)
    # cv2.imwrite("result.png", iamge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.circle(blur, (int(center.y), int(center.x)), 10, (0, 0, 255), 2)
    # cv2.imshow("result", blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
