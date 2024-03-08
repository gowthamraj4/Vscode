import cv2
img = cv2.imread("duplicate.jpg")
grayimg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg",grayimg)
cv2.imshow("image.jpg" ,img)
cv2.imshow("gray.jpg",grayimg)
cv2.waitKey(0)
cv2.displayAllWindows()