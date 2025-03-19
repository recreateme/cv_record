import os
import sys

import cv2
import numpy as np
from scipy.signal import find_peaks


def sobel_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x方向
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y方向
    output = cv2.magnitude(sobel_x, sobel_y)
    return output


def calculate_gradients(arr):
    arr = arr.tolist()
    grad = [0]
    for i in range(1, len(arr) - 1):
        grad.append(abs(arr[i + 1] - arr[i - 1]))
    return np.array(grad)


def find_max_gradient_indices(arr, num_points=4):
    gradients = calculate_gradients(arr)
    max_indices = np.argpartition(gradients, -num_points)[-num_points:]
    sorted_indices = max_indices[np.argsort(-gradients[max_indices])]
    return sorted_indices


def find_pts(array):
    peaks, values = find_peaks(array[1:-1], height=0.2 * np.max(array), distance=15)
    peaks += 1
    peak_values = values['peak_heights']

    assert len(peak_values) >= 4, 'peadks serached is lower than 4'
    top_indices = np.argsort(peak_values)[-4:]
    top_peaks = peaks[top_indices]
    top_peaks = top_peaks[np.argsort(top_peaks)]
    return top_peaks


def calculate_angle(pt1, pt2, pt3):
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def generate_rectangle(points):
    angles = []
    # 计算夹角
    for i in range(4):
        pt1 = points[i]
        pt2 = points[(i + 1) % 4]
        pt3 = points[(i + 2) % 4]
        angle = calculate_angle(pt1, pt2, pt3)
        angles.append((angle, pt1, pt2, pt3))

    # 找到夹角最大的两个边
    max_angle_info = max(angles, key=lambda x: x[0])
    _, pt1, pt2, pt3 = max_angle_info

    # 生成第四个点
    fourth_point = np.array(pt1) + (np.array(pt3) - np.array(pt2))

    return [pt1, pt2, pt3, tuple(fourth_point)]


def get_endpts(rows, cols):
    pts_list = []
    pts_list.append([(rows[0], cols[0]), (rows[0], cols[-1]), (rows[-1], cols[0]), (rows[-1], cols[-1])])
    pts_list.append([(rows[1], cols[1]), (rows[1], cols[2]), (rows[2], cols[1]), (rows[2], cols[2])])
    return pts_list


def find_local_max(mat, points, k=5):
    pts = []
    for point in points:
        x_start = max(point[0] - k, 0)
        x_end = min(point[0] + k + 1, mat.shape[0])
        y_start = max(point[1] - k, 0)
        y_end = min(point[1] + k + 1, mat.shape[1])

        local_area = mat[x_start:x_end, y_start:y_end]
        max_index = np.unravel_index(np.argmax(local_area), local_area.shape)
        row = x_start + max_index[0]
        col = y_start + max_index[1]
        pts.append((row, col))
    return pts


def fine_tune_pts(gradient_image, pts_list):
    for pts in pts_list:
        pts = find_local_max(gradient_image, pts)


def main():
    crop = './analyrect.txt'
    raw = cv2.imread('./overlap.png')
    gray = cv2.imread('./overlap.png', 0)
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    if os.path.exists(crop):
        crop_file = open("analyrect.txt", "r")
        pos = crop_file.readline().strip().split(",")
        pos = [int(x) for x in pos]
        x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
        crop_file.close()
        gray = gray[ y1:y2 + 1, x1:x2 + 1]

        # cv2.rectangle(raw, (y1, x1), (y2, x2), (0, 255, 0), 1)
        # cv2.imshow('raw', raw)
        # cv2.waitKey(0)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # plt.imshow(image, cmap='gray')
    # plt.show()
    edge = sobel_edge_detection(gray)
    row_sum = np.sum(edge, axis=1)
    col_sum = np.sum(edge, axis=0)
    rows = find_pts(row_sum)
    cols = find_pts(col_sum)
    # 将gray转为rgb图像并根据rows和cols画矩形
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # cv2.rectangle(rgb, (cols[0], rows[0]), (cols[-1], rows[-1]), (0, 0, 255), 2)
    # cv2.rectangle(rgb, (cols[1], rows[1]), (cols[2], rows[2]), (0, 255, 0), 2)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pts_list = get_endpts(rows, cols)
    overlap_file_path = os.path.join(current_dir, "overlap")
    f = open(overlap_file_path, "w")
    for pts in pts_list:
        for point in pts:
            f.write(str(point[1]) + "," + str(point[0]) + "\n")
            # cv2.circle(raw, (int(cols[0]), int(rows[0])))
    f.close()


if __name__ == '__main__':
    main()
