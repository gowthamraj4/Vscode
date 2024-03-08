# Importing OpenCV package 
import cv2
import imutils

# Reading the image 
img = cv2.imread('GroupFace.jpg') 

# Converting image to grayscale 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Not required already using grey image

# Loading the required haar-cascade xml classifier file 
haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml') 

# Applying the face detection method on the grayscale image 
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9) 

# Iterating through rectangles of detected faces 
for (x, y, w, h) in faces_rect: 
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	

#resize myself because of image is bigger
resizeImg = imutils.resize(img,width=800)

cv2.imshow('Detected faces', resizeImg) 

cv2.waitKey(10000)
cv2.destroyAllWindows()