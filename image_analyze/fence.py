import os

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def adjusted(mat, up_down=True):
    img = np.array(mat)
    if not up_down:
        img = img.transpose().copy()

    leaf_prof_h = sorted(np.sum(img, axis=0))
    low = np.percentile(leaf_prof_h, 25)
    median = np.percentile(leaf_prof_h, 50)
    high = np.percentile(leaf_prof_h, 75)
    mat = np.array(mat)
    if median * 2 > high + low:
        mat = mat.max() - mat
    return mat


def check_reverse(img, percentiles=[4, 50, 96]):
    img = np.array(img, np.uint8)
    p_low = np.percentile(img, percentiles[0])
    p_mid = np.percentile(img, percentiles[1])
    p_high = np.percentile(img, percentiles[2])
    mid_to_low = abs(p_mid - p_low)
    mid_to_high = abs(p_mid - p_high)
    if mid_to_low > mid_to_high:
        img = -img + img.max() + img.min()
    return img


def findCenter(mat, k=3):
    mat = np.array(mat)
    rows, cols = mat.shape
    assert rows >= k and cols >= k, "传入矩阵小于卷积核"
    # 遍历寻找和最大区域，避免误差中的个别极大值影响整体判断
    max_area = 0
    flag_i = 0
    flag_j = 0
    for i in range(0, rows - k):
        for j in range(0, cols - k):
            totalArea = np.sum(mat[i:i + k, j:j + k])
            if totalArea > max_area:
                max_area = totalArea
                flag_i = i
                flag_j = j

    # 得到最大区域，返回中心点
    return [flag_i + 1, flag_j + 1]


def sobel_edge_detection(image):
    sobel_x = np.array([[-1, 0, 1],
                        [-3, 0, 3],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 3, 1],
                        [0, 0, 0],
                        [-1, -3, -1]])
    height, width = image.shape

    output = np.zeros_like(image)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            gy = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_y)
            output[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    return np.uint8(output)


def horizontal_gradient(image):
    kernel = np.array([1, 0, -1], dtype=np.float32)
    gradient_x = cv2.filter2D(image, ddepth=cv2.CV_32F, kernel=kernel.reshape(1, 3))
    return abs(gradient_x)


def vertical_gradient(image):
    img_t = image.transpose()
    grad_y = horizontal_gradient(img_t).transpose()
    return grad_y


def byside(array):
    left, right = 0, len(array - 1)
    for i in range(1, len(array)):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            left = i
            break
    for j in range(len(array) - 2, left, -1):
        if array[j - 1] < array[j] and array[j] > array[j + 1]:
            right = j
            break
    return left, right


def findMoment(mat, k):
    config = {
        'orientation': "updown",
        "peak_separation": 4,
        "threshold": 0.2,
        "required_prominence": 0.1,
        "fwxm_height": 0.1,
        "min_width": 0.1,
    }
    mat = np.array(mat, dtype=np.float32)
    # hg = horizontal_gradient(mat)
    vg = vertical_gradient(mat)
    # sumOfrow = np.sum(vg, axis=1, keepdims=False)
    edge = sobel_edge_detection(mat)

    col_sum = np.sum(edge, axis=0, keepdims=False)
    row_sum = np.sum(edge, axis=1, keepdims=False)
    ratio = 1
    num = 0
    while True:
        if num > 20:
            # sort_idx = np.argsort(col_sum)
            # peak_col_idxs = sorted(sort_idx[:-3:-1])
            peak_col_idxs = byside(col_sum)
            break
        num += 1
        peak_col_idxs, peak_props = find_peaks(
            col_sum,
            rel_height=(1 - config["fwxm_height"]),
            width=k,
            height=np.percentile(col_sum, 60),
            distance=3,
            prominence=config["required_prominence"] * ratio,
        )
        if len(peak_col_idxs) < 2:
            ratio *= 0.8
        elif len(peak_col_idxs) > 2:
            ratio *= 1.5
        else:
            break
    ratio = 1
    num = 0
    while True:
        if num > 20:
            # sort_idx = np.argsort(row_sum)
            # peak_row_idxs = sorted(sort_idx[:-3:-1])
            peak_row_idxs = byside(row_sum)
            break
        num += 1
        peak_row_idxs, peak_props = find_peaks(
            row_sum,
            rel_height=(1 - config["fwxm_height"]),
            width=k,
            height=np.percentile(row_sum, 60),
            distance=3,
            prominence=config["required_prominence"],
        )
        if len(peak_row_idxs) < 2:
            ratio *= 0.8
        elif len(peak_row_idxs) > 2:
            ratio *= 1.5
        else:
            break
    if peak_col_idxs[0] + 1 <= peak_col_idxs[1] or peak_row_idxs[0] + 1 <= peak_row_idxs[1]:
        pass
    return peak_row_idxs[0], peak_row_idxs[1], peak_col_idxs[0], peak_col_idxs[1]


def process_peak_ids(peak_ids, error_permit):
    peak_ids = np.array(peak_ids)
    diff = np.diff(peak_ids)
    median_diff = np.max(diff)
    if abs(median_diff - diff.max()) < error_permit and abs(median_diff - diff.min()) < error_permit:
        return "ok", None

    # 判断处理其他情况
    keep = []
    waiting_next = False
    processed_spaces = []
    start_space = None
    tp = []
    for i in range(len(diff)):
        if waiting_next == True:
            if abs(sum(tp) + diff[i] - median_diff) <= error_permit:
                keep.append(i)
                start_space = None
                tp = []
                waiting_next = False
                continue
            elif median_diff - sum(tp) - diff[i] > error_permit:
                tp.append(diff[i])
                continue
            else:
                return "less", None
        if abs(diff[i] - median_diff) <= error_permit:
            keep.append(i)
            continue
        if diff[i] - median_diff > error_permit:
            return "less", None
        if diff[i] - median_diff < error_permit:
            start_space = i
            tp.append(diff[i])
            waiting_next = True
    # print(f"keep:{keep}")
    # keep.append(keep[-1] + 1)
    positions = []
    # for k in keep:
    #     positions.extend(peak_ids[k:k+2])
    # positions去重
    keep = np.array(keep) + 1
    idx = [0] + list(keep)
    positions = [peak_ids[x] for x in idx]
    return "changed", positions


def findPeak(values, threshold, config, k_min=2, k_max=50, vertical=1):
    config = {
        'orientation': "updown",
        "peak_separation": 4,
        "threshold": 0.2,
        "required_prominence": 0.1,
        "fwxm_height": 0.1,
        "min_width": 0.1,
    }
    num = 0
    r_prominence, r_height, r_rel_height = 1, 1, 1
    dis_shift = 0
    while True:
        if num > 100:
            raise "迭代溢出"
        num += 1
        idxs, props = find_peaks(
            values,
            rel_height=(1 - config["fwxm_height"]) * r_rel_height,
            width=config["min_width"] + dis_shift,
            height=threshold * r_height,
            distance=config["peak_separation"],
            prominence=config["required_prominence"] * r_prominence * vertical,
        )
        if len(idxs) < k_min:
            # r_height *= 0.9
            r_prominence *= 0.8
            continue
            # r_rel_height *= 1.1

        elif len(idxs) >= k_max:
            # r_height *= 1.2
            r_prominence *= 1.2
            continue

        status, positions = process_peak_ids(idxs, 6)
        if status == "ok":
            return idxs, props
        elif status == "less":
            r_prominence *= 0.8
            continue
        elif status == "changed":
            if k_min < len(positions) < k_max:
                idxs = positions
                return idxs, props
            else:
                r_prominence *= 0.8
                continue
    return idxs, props


def find_fence(image):
    row, col = image.shape
    image = np.array(image, dtype=np.float32)
    image = cv2.GaussianBlur(image, (5, 5), 3, sigmaY=0)

    profile = np.sum(image, axis=0, dtype=np.float32)
    # visualize(profile, "profile")

    g_left = profile[1:] - profile[:-1]
    g_right = profile[:-1] - profile[1:]
    idx1 = np.argmax(g_left)
    idx2 = np.argmax(g_right)
    return idx1 + 1, idx2


def find_fence2(image):
    m, n = image.shape
    min_diff = float('inf')
    best_k1, best_k2 = None, None

    for k1 in range(1, n):
        for k2 in range(k1 + 1, n - 1):
            # 计算三个区域的均值
            sum_left = np.sum(image[:, :k1])
            sum_right = np.sum(image[:, k2:])
            mean_middle = np.mean(image[:, k1:k2 + 1])
            back_mean = (sum_left + sum_right) / (m * (n + k1 - k2 + 1))
            # 计算差值
            diff = back_mean - mean_middle
            # 更新最佳分割点
            if diff < min_diff:
                min_diff = diff
                best_k1, best_k2 = k1, k2
    return best_k1, best_k2


def visualize(arr, name="arr"):
    plt.plot(arr)
    plt.title(name)
    plt.show()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    crop_file = "analyrect.txt"
    img_name = 'fence.png'
    img = cv2.imread(img_name, 0)
    # cv2.imshow("raw", img)
    # cv2.waitKey(0)
    if os.path.exists(crop_file):
        crop_file = open("analyrect.txt", "r")
        pos = crop_file.readline().strip().split(",")
        pos = [int(x) for x in pos]
        x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
        crop_file.close()
        img = img[y1:y2 + 1, x1:x2 + 1]
        # cv2.imshow("crop", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # 图片路径
    # 读取图像并进行CLAHE处理
    row, col = img.shape
    # img = cv2.resize(img, (row*2, col*2), interpolation=cv2.INTER_CUBIC)
    adjusted_img = adjusted(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    image = clahe.apply(adjusted_img)

    # cv2.imwrite("clahe.png", image)
    # cv2.imshow("clahe", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    config = {
        'orientation': "updown",
        "peak_separation": 4,  # 峰值分离，单位：像素
        "threshold": 0.2,  # 阈值，调整为0.2以识别更多峰值
        "required_prominence": 0.1,  # 峰值显著性，调整为0.1以识别更多峰值
        "fwxm_height": 0.1,  # FWXM高度
        "min_width": 0.1,  # 最小宽度，调整为0.01以识别更窄的峰值
    }
    leaf_prof = np.mean(image, 0, keepdims=False)
    values = leaf_prof / leaf_prof.max()
    range_values = values.max() - values.min()
    threshold = values.min() + config["threshold"] * range_values
    # visualize(values)
    peak_idxs, peak_props = findPeak(values, threshold, config)

    padding = (peak_idxs[1] - peak_idxs[0]) // 2
    leaf_prof_h = np.mean(image[:, peak_idxs[1] - padding:peak_idxs[1] + padding], 1, keepdims=False)
    # leaf_prof_h = np.sum(image, axis=1, keepdims=False)
    values_h = leaf_prof_h / leaf_prof_h.max()  # 归一化
    range_values_h = values_h.max() - values_h.min()
    threshold_h = values_h.min() + config["threshold"] * range_values_h
    # visualize(values_h, "values_h")
    peak_idxs_h, peak_props_h = findPeak(values_h, threshold_h, config, vertical=0.2)
    # peak_values = values_h[peak_idxs_h]
    # f.write(f"{peak_idxs_h.shape[0]},{peak_idxs.shape[0]}\n")
    # visualization (for test, Comment the segment)
    # import matplotlib.pyplot as plt
    #
    # num = np.arange(len(values))
    # value = np.ravel(values_h)
    # plt.plot(num, value, color="red")
    # plt.scatter(peak_idxs_h, peak_values)
    # plt.show()

    cross_points = [[(x, y) for y in peak_idxs] for x in peak_idxs_h]
    # 测试
    # 可视化交叉点
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # # 计算交叉点
    # for line in cross_points:
    #     for x, y in line:
    #         cv2.circle(image, (y, x), 5, (0, 0, 255), -1)
    # cv2.imwrite("pts.png", image)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    row, col = image.shape
    row_spacing = max(int((peak_idxs_h[1] - peak_idxs_h[0]) // 8), 3)
    col_spacing = int((peak_idxs[1] - peak_idxs[0]) // 2.5)
    # print(peak_idxs_h)
    # print(col_spacing)
    widthes = []

    lines = []
    line = []
    if peak_idxs_h[0] > 0.5 * col_spacing:
        for point in cross_points[0]:
            left = 0 if point[1] - col_spacing < 0 else point[1] - col_spacing
            right = col - 1 if col - 1 < point[1] + col_spacing else point[1] + col_spacing
            local_area = image[0:point[0], left:right]
            # cv2.rectangle(tp, (left, 0), (right, point[0] - 2), (0, 0, 255), 1)
            # cv2.imshow("image", tp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            c1, c2 = find_fence(local_area)
            line.append((0, point[0] - 2, left + c1, c2 + left))
            widthes.append(c2 - c1 + 1)
        lines.append(line)

    # line = []
    # for point in cross_points[-1]:
    #     left = 0 if point[1] - col_spacing < 0 else point[1] - col_spacing
    #     right = col - 1 if col - 1 < point[1] + col_spacing else point[1] + col_spacing
    #     local_area = image[point[0]:row, left:right+1]
    #     c1, c2 = find_fence(local_area)
    #     line.append((0, point[0] - 2, left + c1, c2 + left))
    # lines.append(line)

    for i in range(len(cross_points) - 1):
        line = []
        for j in range(len(cross_points[i])):
            tp = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            point = cross_points[i][j]
            if i == len(cross_points) - 1:
                point2 = [row - 1, point[1]]
            else:
                point2 = cross_points[i + 1][j]
            left = 0 if point[1] - col_spacing < 0 else point[1] - col_spacing
            right = col - 1 if col - 1 < point[1] + col_spacing else point[1] + col_spacing
            ceil = point[0] + row_spacing
            floor = point2[0] - row_spacing + 1
            local_area = image[ceil:floor, left:right + 1]
            c1, c2 = find_fence(local_area)

            # cv2.rectangle(tp, (left, ceil),(right, floor), (0, 0, 255), 1)
            # cv2.rectangle(tp, (left+c1, ceil),(left+c2, floor), (0, 255, 0), 1)
            # cv2.imshow("image", tp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            line.append((ceil, floor, left + c1, c2 + left))
            widthes.append(c2 - c1 + 1)
        lines.append(line)
    # 可视乎
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # for line in lines:
    #     for point in line:
    #         cv2.rectangle(img, (point[2], point[0]), (point[3], point[1]), (0, 0, 255), 1)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    widthes = np.array(widthes).reshape(-1, len(peak_idxs))
    median = np.median(widthes)
    # print(widthes)
    # print(f'median: {median}')

    baselines = []
    for num in peak_idxs:
        right = num + median // 2
        baselines.append((int(max(0, 2 * num - right)), int(min(col - 1, right))))
    # 写入文件
    # print(f'baselines: {baselines}')
    f = open("fence", "w")
    f.write(f"{len(peak_idxs)},{len(lines)}\n")
    for line in lines:
        for j in range(len(line)):
            # 观测
            num1 = (line[j][0], line[j][2], line[j][1], line[j][3])
            num1 = ",".join(str(elemnet) for elemnet in num1)
            f.write(f"{num1}\n")
            # 标准
            num2 = (line[j][0], baselines[j][0], line[j][1], baselines[j][1])
            num2 = ",".join(str(elemnet) for elemnet in num2)
            f.write(f"{num2}\n")

    f.close()


if __name__ == '__main__':
    main()

    # 使用os库创建文件夹，存在就不创建
    # os.makedirs("执行完毕", exist_ok=True)

    # 结束测试
    # roi region
    # col_spacing = int((peak_idxs_h[1] - peak_idxs_h[0]) // 4)
    # row_spacing = col_spacing
    #
    # # iterate each fence
    # lines = []
    # for line in cross_points:
    #     X1, X2, Y1, Y2 = [], [], [], []
    #     for x, y in line:
    #         # expand to find central area
    #         local_area = image[x - col_spacing:x + col_spacing + 1, y - row_spacing:y + row_spacing + 1]
    #         local_x, local_y = findCenter(local_area)
    #         x = x - col_spacing + local_x
    #         y = y - row_spacing + local_y
    #
    #         local_area = image[x - col_spacing:x + col_spacing + 1, y - row_spacing:y + row_spacing + 1]
    #         x1, x2, y1, y2 = findMoment(local_area, k=3)
    #
    #         x1 += x - col_spacing
    #         x2 += x - col_spacing
    #         y1 += y - row_spacing
    #         y2 += y - row_spacing
    #
    #         X1.append(x1)
    #         X2.append(x2)
    #         Y1.append(y1)
    #         Y2.append(y2)
    #     lines.append([X1, X2, Y1, Y2])
    # lines = np.array(lines)
    # # 测试：
    # # window = cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # # for line in lines:
    # #     for j in range(line.shape[1]):
    # #         cv2.rectangle(image, (line[0, j], line[2, j]), (line[2, j], line[3, j]), (0, 0, 255), -1)
    # # cv2.imshow("image", image)
    # # cv2.waitKey(0)
    # # 测试结束
    #
    # # for i in range(1,len(lines)):
    # #     lines[i][2:4, :] = lines[i-1][2:4, :]
    #
    # # set criterion
    # dis = lines[:, 1, :] - lines[:, 0, :]
    # dis_row = round(np.sum(dis) / dis.size)
    # dis2 = lines[:, 3, :] - lines[:, 2, :]
    # dis_col = round(np.sum(dis2) / dis2.size)
    # # print(dis_col)
    #
    # # align all the rec
    # # print(lines[0])
    # # print("=" * 88)
    #
    # for i in range(len(lines)):
    #     # row align
    #     line = lines[i]
    #     arr0 = line[0]
    #     arr1 = line[1]
    #     # counts = np.bincount(arr0)
    #     # mode0 = np.argmax(counts)
    #     # counts = np.bincount(arr1)
    #     # mode1 = np.argmax(counts)
    #     # lines[i][0] = mode0
    #     # lines[i][1] = mode1
    #     lines[i][0] = np.median(arr0)
    #     lines[i][1] = np.median(arr1)
    #
    # #     # col align
    # #     for j in range(line.shape[1]):
    # #         if line[3, j] - line[2, j] != dis_col:
    # #             # adjust left
    # #             left_index = line[3, j] - dis_col
    # #             # adjust right
    # #             right_index = line[2, j] + dis_col
    # #
    # #             left_area = image[mode0:mode1 + 1, left_index:line[3, j] + 1]
    # #             right_area = image[mode0:mode1 + 1, line[2, j]:right_index + 1]
    # #             left_sum = np.sum(left_area)
    # #             right_sum = np.sum(right_area)
    # #             if left_sum >= right_sum:
    # #                 lines[i][2][j] = left_index
    # #             else:
    # #                 lines[i][3][j] = right_index
    #
    # # print(lines[0])
    #
    # # 横向
    # # for line in lines:
    # #     for j in range(line.shape[1]):
    # #         cv2.rectangle(image, (line[2, j], line[0, j]), (line[3, j], line[1, j]), (0, 0, 255), -1)
    # # cv2.imwrite("result.jpg", image)
    # # cv2.imshow("image", image)
    #
    # # 可视化
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # f1.write(f"{len(peak_idxs)},{len(peak_idxs_h)-1}\n")
    # # f2.write(f"{len(peak_idxs)},{len(peak_idxs_h)}\n")
    # for i in range(len(lines) - 1):
    #     mean_length = 0
    #     for j in range(lines[i].shape[1]):
    #         mean_length += (lines[i][3][j] - lines[i][2][j])
    #     mean_length = round(mean_length / lines[i].shape[1])
    #
    #     c1 = None
    #     c2 = None
    #     if mean_length % 2 == 1:
    #         # 如果基准长度为奇数，直接设置
    #         c1 = np.array(peak_idxs) - mean_length // 2
    #         c2 = np.array(peak_idxs) + mean_length // 2
    #     else:
    #         # 如果标砖长度为偶数，
    #         c1 = np.array(peak_idxs) - mean_length // 2 - 1
    #         c2 = np.array(peak_idxs) + mean_length // 2
    #
    #     for j in range(lines[i].shape[1]):
    #         x1 = lines[i][1, j]
    #         x2 = lines[i + 1][0, j]
    #         y1 = lines[i][2, j]
    #         y2 = lines[i][3, j]
    #         cv2.rectangle(image, (y1, x1), (y2, x2), (0, 0, 255), -1)
    #         # width = lines[i + 1][0, j] - lines[i][1, j]
    #         # width = x2 - x1
    #
    #         left = int(peak_idxs[j] - mean_length // 2)
    #         right = int(2 * peak_idxs[j] - left)
    #         f1.write(f"{x1},{y1},{x2},{y2}\n")
    #         f1.write(f"{x1},{c1[j]},{x2},{y2}\n")
    #         # print(f"width: {width}")
    #     # print("=" * 99)
    # # 画基线
    # for i in range(len(peak_idxs)):
    #     cv2.line(image, (peak_idxs[i], 0), (peak_idxs[i], image.shape[0]), (255, 0, 0), 1)
    #     # pass
    #
    # # statistics
    # # f1.write(f"{",".join([str(x) for x in peak_idxs])}\n")
    # # f1.write(f"{",".join([str(x) for x in peak_idxs_h])}")
    # # f.close()
    # f1.close()
    # # f2.close()
    # # cv2.imwrite("result.jpg", image)
    # # cv2.imshow("image", image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
