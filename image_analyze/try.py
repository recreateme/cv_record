import cv2
import matplotlib.pyplot as plt


img = cv2.imread('star.png', 0)

# _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow("binary", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 使用matplotlib绘制直方图
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()