import cv2
img = cv2.imread("Resize.jpg")
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
## 0 black , 255 white 0 - 120 will be black
thershold = cv2.threshold(gray, 121 , 255, cv2.THRESH_BINARY) [1]
cv2.waitKey(0)
cv2.imshow("gray",gray)
cv2.imshow("thershold" ,thershold )
cv2.waitKey(0)
cv2.displayAllWindows()