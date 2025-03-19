import cv2

lines = open("star", "r").readlines()
result = lines[0].strip().split(",")
result = [int(float(x)) for x in result]

offset = open("analyrect.txt","r").readline().strip().split(',')
offset = [int(x) for x in offset]

print("offset",offset)
print("result",result)

img = cv2.imread("star.png", 1)
for i in range(2, len(lines)):
    line = lines[i].strip().split(",")[0:-1]
    line = [int(float(x)) for x in line]
    cv2.line(img, (line[0]+offset[0], line[1]+offset[1]), (line[2]+offset[0], line[3]+offset[1]), (0,255,0), 2)


cv2.circle(img, (result[0]+offset[0], result[1]+offset[1]), 10,(255,0,0), 2)

cv2.imshow("raw", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
