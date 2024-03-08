import cv2
import numpy as np

# Create a black image
#Creating a NumPy array representing an image with dimensions 300 pixels in height, 500 pixels in width, and 3 channels 
img = np.zeros((300, 500, 3), dtype=np.uint8) #(number of color channels (BGR in this case))

# Define text and its properties
text = "Hello, OpenCV!"
org = (50, 150)          # Position # coordinates of the bottom-left corner of the text string in the image
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1            # Size # multiplied by the font-specific base size
color = (255, 255, 255)  # White color in BGR
thickness = 2
lineType = cv2.LINE_AA

# Put the text on the image
cv2.putText(img, text, org, font, fontScale, color, thickness, lineType)

# Display the image
cv2.imshow('Image with Text', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()