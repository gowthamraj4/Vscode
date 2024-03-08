import numpy as np #math
import imutils #resize the image
import cv2 #image acq.
import time #delay

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
#Appling minimum thersold value to find the object,
# minimum the thersold value higher the chance to detect the object, not find or recoginise
confThresh = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# Appling random color to the labels
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
## is used to load a deep neural network (DNN) model from two files:
## a Caffe prototxt file (.prototxt) that defines the architecture of the neural network, and
## a Caffe model file (.caffemodel) that contains the pre-trained weights of the network. ##
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
print("Starting Camera Feed...")
cam = cv2.VideoCapture(0) #camera init.
time.sleep(2.0)

while True:
	# (boolean , numpy.ndarray) - numpy.ndarray is a multi-dimensional array contains the pixel values of the frame.
	_,frame = cam.read() #reading frame from the camera

	frame = imutils.resize(frame, width=500) #resize the frame to be displayed as window

	(h, w) = frame.shape[:2] # .shape returns a tuple with three elements (height, width, channels):
	# frame.shape[:2] -> start:stop

	#preprocessing
	imResize = cv2.resize(frame, (300, 300)) #resize

	#Syntax
	#blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
	blob = cv2.dnn.blobFromImage(imResize,0.007843, (300, 300), 127.5) #blobed image


	#net = cv2.dnn.readNetFromCaffe(prototxt, model)

	#After setting the input using setInput, calling forward runs the forward pass of the neural network.
	#It computes the output of the network based on the input data. 

	net.setInput(blob) #set the blobbed image as input
	
	detections = net.forward() #passing pre processed image into model
	print(detections)

	# The return type of .forward() is a NumPy array (numpy.ndarray).
	# .shape  will return the shape of the array. The shape contains information about the dimensions of the output
	# .shape[2] The return type of this expression is an integer , Specifically the size (number of elements) along the third dimension of the array.

	#detections.shape[2] extracts the size (number of elements) along the third dimension,
	# which corresponds to the height or, more specifically, the number of cv2.dnn.readNetFromCaffe(prototxt, model).forward().shape[2]  or features in each frame.
	detShape = detections.shape[2]
	detectionShape = cv2.dnn.readNetFromCaffe(prototxt, model).forward().shape[2]
	print('detections Shape :',  detectionShape)
	#The shape of this array is typically organized as (batch_size, number_of_channels, height, width).


	# The detShape variable is an integer representing the size of the third dimension array
	# (likely the output of a neural network forward pass).
	
	# numpy.arange([start, ]stop, [step, ], dtype=None) 
	for i in np.arange(0,detShape): # start from 0 to { cv2.dnn.readNetFromCaffe(prototxt, model).forward().shape[2] }

		confidence = detections[0, 0, i, 2] # 2 - confident level
		if confidence > confThresh:

			idx = int(detections[0, 0, i, 1]) # 1 - id
			#print("ClassID:",detections[0, 0, i, 1])

			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

			# np.array([xmin, ymin, xmax, ymax])
			# h, w = frame.shape[:2]
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			# print("boxCoord:",detections[0, 0, i, 3:7]) - co-ordinate

			(startX, startY, endX, endY) = box.astype("int")
			
			
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			
			# for automattically adjusting the label up & down on the box
			
			if startY - 15 > 15:
				y = startY - 15 # startY -  y-coordinate of the top-left corner of the bounding box.
			else:
				y = startY + 15 

			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
cam.release()
cv2.destroyAllWindows()