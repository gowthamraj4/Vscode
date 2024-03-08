import cv2
img = cv2.imread("Resize.jpg")
blur1= cv2.GaussianBlur(img,(21,21),0)
blur2 = cv2.GaussianBlur(img,(5,5),5)
cv2.imshow("blur1", blur1)
cv2.imshow("blur2", blur2)
cv2.waitKey(0)