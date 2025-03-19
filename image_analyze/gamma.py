import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class EllipseDetector:
    def __init__(self, filename, txt=None):
        self.txt = txt
        self.filename = filename

        self.raw_image = self.get_raw_image()
        self.imgs = self.crop_imgs()

        self.imgs = self.preprocess_images()
        self.ellipses = self.detect_ellipse()

        # cv2.imshow('img', self.imgs[0])
        # cv2.waitKey(0)

    def crop_imgs(self):
        if self.txt is None:
            return [self.raw_image]
        imgs = []
        with open(self.txt) as f:
            lines = f.readlines()
            for line in lines:
                pos = line.strip().split(",")
                pos = [int(x) for x in pos]
                x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
                image = self.raw_image[y1:y2 + 1, x1:x2 + 1]
                # cv2.imshow('crop1', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                imgs.append(image)
            return imgs

    def get_raw_image(self):
        return cv2.imread(self.filename, 0)

    def preprocess_images(self):
        imgs = []
        for img in self.imgs:
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            imgs.append(blurred)
        return imgs

    def detect_edges(self, image, method='otsu'):
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        if method == 'adaptive':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # cv2.imshow('img', binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return binary

    def find_ellipse_contours(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def fit_ellipses(self, contours, min_size=20):
        ellipses = []
        for contour in contours:
            # 轮廓点数要足够多才能拟合椭圆
            if len(contour) >= 5:
                area = cv2.contourArea(contour)
                if area > min_size:
                    ellipse = cv2.fitEllipse(contour)
                    return ellipse

        raise ValueError("Not Found")

    def stats(self, binary, x1, y1, x2, y2):
        image = binary[y1:y2 + 1, x1:x2 + 1]
        median = np.median(image)

        array = (image[image.shape[0] // 2, :] - median).tolist()
        for i, val in enumerate(array):
            if val <= 0:
                left = i
                break
        for i, val in enumerate(array[::-1]):
            if val <= 0:
                right = len(array) - i - 1
                break
        array = (image[:, image.shape[1] // 2] - median).tolist()
        for i, val in enumerate(array):
            if val <= 0:
                ceil = i
                break
        for i, val in enumerate(array[::-1]):
            if val <= 0:
                floor = len(array) - i - 1
                break
        d1 = right - left
        d2 = floor - ceil
        return min(d1, d2)

    def detect_ellipse(self):
        if len(self.imgs) == 0:
            raise ValueError("No image found.")
        ellipses = []
        for img in self.imgs:
            binary = self.detect_edges(img, method='otsu')
            # cv2.imshow('binary', binary)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            contours = self.find_ellipse_contours(binary)

            max_contour = max(contours, key=lambda contour: contour.shape[0])
            ellipse = self.fit_ellipses([max_contour])
            # print(ellipse)

            center, axes, angle = ellipse
            center = [int(x) for x in center]
            radius = int((axes[0] + axes[1]) // 4)

            x1, y1 = (center[0] - radius, center[1] - radius)
            x2, y2 = (center[0] + radius, center[1] + radius)
            diameter = self.stats(img, x1, y1, x2, y2)
            ellipses.append([center, diameter])

            # 可视化
            # bag = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # cv2.rectangle(bag, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # cv2.circle(bag, center, int(diameter / 2), (0, 255, 0), 1)
            # cv2.circle(bag, center, 5, (0, 255, 0), 1)
            # cv2.imwrite("bag.png", bag)
            # cv2.imshow('bag', bag)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return ellipses

    def writeToFile(self, f):
        for ellipse in self.ellipses:
            center, diameter = ellipse
            center = [int(x) for x in center]
            f.write(str(center[0]) + ',' + str(center[1]) + ',' + str(diameter) + "," + str(diameter) + '\n')
        f.close()


def main():
    filename = './gamma.png'
    txt = './analyrect.txt'
    f = open('./gamma', 'w')
    if not os.path.exists(txt):
        txt = None
    detector = EllipseDetector(filename, txt)
    detector.writeToFile(f)

    # 可视化
    # img = cv2.imread(filename)
    # center, axes, angle = detector.ellipse
    # center = [int(x) for x in center]
    # radius = int((axes[0] + axes[1]) // 4)
    # cv2.circle(img, (center[0], center[1]), radius, (0, 0, 255), 1)
    # cv2.imshow('可视化', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(detector.ellipse)


if __name__ == "__main__":
    main()
