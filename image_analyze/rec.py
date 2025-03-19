import os
import time

import cv2
import numpy as np


def Length(A, B):
    A = np.array(A)
    B = np.array(B)
    return np.linalg.norm(A - B)


def find_point(C, D, length_AD, big_flag=True):
    C = np.array(C)
    D = np.array(D)
    CD = D - C

    perp_CD = np.array([-CD[1], CD[0]])  # 交换x和y分量，并改变一个分量的符号
    unit_perp_CD = perp_CD / np.linalg.norm(perp_CD)

    A1 = C + unit_perp_CD * length_AD
    A2 = C - unit_perp_CD * length_AD

    if big_flag:
        return A1 if A1[1] > A2[1] else A2
    else:
        return A1 if A1[1] < A2[1] else A2


def calculate_fourth_point(pts1):
    pts = np.float32(pts1)
    sorted_indices = np.argsort(pts[:, 0])
    pts = pts[sorted_indices]

    left = pts[:2, :]
    right = pts[2:, :]

    # 排序对齐
    sorted_indices = np.argsort(left[:, 1])
    left = left[sorted_indices]
    sorted_indices = np.argsort(right[:, 1])
    right = right[sorted_indices]
    A = left[0]
    D = left[1]
    B = right[0]
    C = right[1]

    angle = 0
    if A[0] == D[0]:
        mean = (B[0] + C[0]) // 2
        B[0] = C[0] = mean
        B[1] = A[1]
        C[1] = D[1]
    elif B[0] == C[0]:
        mean = (A[0] + D[0]) // 2
        A[0] = D[0] = mean
        A[1] = B[1]
        D[1] = C[1]
    elif A[1] == B[1]:
        mean = (C[1] + D[1]) // 2
        C[1] = D[1] = mean
        C[0] = B[0]
        D[0] = A[0]
    elif C[1] == D[1]:
        mean = (A[1] + B[1]) // 2
        A[1] = B[1] = mean
        B[0] = C[0]
        A[0] = D[0]
    else:
        # 不存在垂直边
        AB = B - A
        CD = D - C
        theta_AB = np.arctan2((B[1] - A[1]), (B[0] - A[0]))
        theta_CD = np.arctan2((C[1] - D[1]), (C[0] - D[0]))
        width = (Length(A, D) + Length(B, C)) // 2
        if abs(theta_AB) > abs(theta_CD):
            length = Length(C, D)
            flag_AB_base = 0
            theta = theta_CD
        else:
            length = Length(A, B)
            flag_AB_base = 1
            theta = theta_AB
        if flag_AB_base == 0:
            A1 = find_point(D, C, width, big_flag=False)
            A = [round(x) for x in A1]
            B1 = find_point(C, D, width, big_flag=False)
            B = [round(x) for x in B1]
        else:
            C1 = find_point(B, A, width, big_flag=True)
            C = [round(x) for x in C1]
            D1 = find_point(A, B, width, big_flag=True)
            D = [round(x) for x in D1]

    return angle, np.uint32([A, B, C, D])


def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    return np.arccos(cos_theta) * 180 / np.pi


def sobel_edge_detection(image):
    image = np.array(image)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    height, width = image.shape

    output = np.zeros_like(image)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # 计算x和y方向梯度
            gx = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            gy = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_y)
            output[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    return output


def find_largest_rectangle(magnitude):
    _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow('1', erosion)
    # cv2.waitKey(0)
    if erosion[0, 0] > 0 or erosion[0, -1] > 0:
        erosion = 255 - erosion
    # cv2.imshow('2', erosion)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    largest_rectangle = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_rectangle = approx

    # 返回矩形的四个顶点坐标
    if largest_rectangle is not None:
        return largest_rectangle.reshape(4, 2)
    else:
        return None


def main():
    img_path = 'rec.png'
    img = cv2.imread(img_path, 0)

    crop_file = "analyrect.txt"
    if os.path.exists(crop_file):
        crop_file = open("analyrect.txt", "r")
        pos = crop_file.readline().strip().split(",")
        pos = [int(x) for x in pos]
        x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
        crop_file.close()
        img = img[y1:y2 + 1, x1:x2 + 1]

    magnitude = img
    largest_rectangle = find_largest_rectangle(magnitude)

    # 可视化
    # if largest_rectangle is not None:
    #     # img_with_rectangle = cv2.drawContours(img, [largest_rectangle.astype(int)], 0, (0, 255, 0), 2)
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     cv2.rectangle(img, largest_rectangle[0], largest_rectangle[2], (0, 255, 0), 2)
    #     cv2.imshow("result", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    angle, pts = calculate_fourth_point(largest_rectangle)

    f = open("rec", "w", encoding="utf-8")
    f.write(f"{angle}\n")
    for point in pts:
        f.write("{},{}\n".format(point[0], point[1]))

    f.close()


if __name__ == '__main__':
    main()