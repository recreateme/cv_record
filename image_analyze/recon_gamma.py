import cv2

result = open("gamma", "r").readline().strip().split(",")
result = [int(x) for x in result]

offset = open("analyrect.txt","r").readline().strip().split(',')
offset = [int(x) for x in offset][:2]

print("offset",offset)
print("result",result)

img = cv2.imread("gamma.png", 1)

cv2.circle(img, (result[0]+offset[0], result[1]+offset[1]), result[-1]//2, (0,0,255), 2)

cv2.imshow("plot", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
