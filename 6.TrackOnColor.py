import imutils #resizing library - import
import cv2 #opencv


#hsv color format values

#Camera init.

while True:

        #reading frame from camera

        #resizing the Frame
        
        #smoothing
        
        #convert bgr to hsv color format

        #mask the green color

        #erode
        
        #dilate
                
                center = None

        #find cnts
        

                # Draw Minimum EnclosingCircle , obtain x, y and radius from the circle

                #find momentum , to find the area of center

                if radius > 10:
                        # draw circle

                        # draw center of circle

                        #print(center,radius)

                        if radius > 250:
                                print("stop")
                        else:
                                if(center[0]<150):
                                        print("Left")
                                elif(center[0]>450):
                                        print("Right")
                                elif(radius<250):
                                        print("Front")
                                else:
                                        print("Stop")
        #imshow
        #waitkey
        

#release camera

#close windows
