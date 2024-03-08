import cv2
import imutils
img = cv2.imread("gray.jpg")
resizeImg = imutils.resize(img,width=200)
cv2.imwrite("Resize.jpg" , resizeImg)