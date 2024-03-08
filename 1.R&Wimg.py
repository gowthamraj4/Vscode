import cv2
img = cv2.imread("image.jpg")
cv2.imwrite("duplicate.jpg" ,img)
cv2.imshow("Copy" , img)
cv2.waitKey(0)
cv2.destroyAllWindows()