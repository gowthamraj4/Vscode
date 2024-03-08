import cv2 #handling images
import numpy #array
import os #handling a directories

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file) #loading face detector
datasets = 'dataset'

print('Training...')
(images, labels, names, id) = ([], [], {}, 0) # empty lists and directories

# subdirs - dataset
# dirs - list of all subdirectories
# Eg : dataset / sanjay , dataset / arun 

for (dirs, subdirs , files) in os.walk(datasets):
# subdirs returs list of strings , subdir stores one string value
   for subdir in subdirs:
        names[id] = subdir # subdir store one subdirectory name in {} list
        subjectpath = os.path.join(datasets, subdir) #subdir = names[0]
        # subjectpath : dataset\elon , scope is with all files in elon 1.png to 30.png
        print('subjectpath :' + subjectpath)

        for filename in os.listdir(subjectpath): # subjectpath returs list of values , filename stores one string value
            # filename : 1.png
            path = subjectpath + '/' + filename # path : dataset\elon/1.png ..
            label = id # 0
            images.append(cv2.imread(path, 0)) # returns 
            labels.append(int(label))    
        id +=1

#for lis in [images, labels]:
   # (images, labels) = numpy.array(lis)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)                   

model = cv2.face.LBPHFaceRecognizer_create() #loading face recognizer
#model =  cv2.face.FisherFaceRecognizer_create()

model.train(images, labels) #training dataset

webcam = cv2.VideoCapture(0) 
cnt=0

while True:
    (_, im) = webcam.read()
    #_,im = cv2.imread(webcam)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #face detection
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        (width, height) = (130, 100)
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize) #predict/classify face
        # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) dont know the function

        if prediction[1]<800:
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255),2)
            print (names[prediction[0]])
         #   cnt=0 , commanded because dont know the function
        else:
            cnt+=1
            cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("unKnown.jpg",im)
                cnt=0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()