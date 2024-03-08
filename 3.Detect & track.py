import cv2 #image
import time #delay
import imutils #resize

cam = cv2.VideoCapture(0) #cam id
time.sleep(1)

firstFrame=None
area = 500

while True:
    _,img = cam.read() #read frame from camera
    text = "Normal" 
    img = imutils.resize(img, width=500) #resize
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color 2 Gray scale image
    
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0) #smoothened
    
    #run only once
    if firstFrame is None: #2ND time firstframe is not none , it's a stored value 
            firstFrame = gaussianImg #capturing 1st frame on 1st iteration
            continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg) #absolute diff b/w 1st nd current frame , original - current = imgDiff
    
    ## 0 black , 255 white 0 - 25 will be black
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #thresholding on imgdiff image
    
    threshImg = cv2.dilate(threshImg, None, iterations=2) # remove holes , leftovers
    
    # fill the waterbottle areas created by dilate method and store in cnts
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine all neighborhood PIXEL of an object, to make an area 
    cnts = imutils.grab_contours(cnts) # grab total area

    for c in cnts:
            if cv2.contourArea(c) < area: # if object is moving , continuously measures the area 
                    continue
            (x, y, w, h) = cv2.boundingRect(c) # continously provide th x,y , wide & height of the moving object 
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # continuouly draw rectangle
            text = "Moving Object detected"
    print(text)
    cv2.putText(img, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
