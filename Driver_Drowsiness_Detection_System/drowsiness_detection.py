import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('alarm.wav')

# HAAR Cacade: Algorithm to detect objects, faces, outlines etc
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml') # to detect faces
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml') # to detect left eye
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml') # to detect right eye

lbl = "Closed"

model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0 # score to check how long the eyes are closed
thicc = 2 # thickness of rectangle
rpred = [99]
lpred = [99]

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Get image from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting face, left and right eye using Haar Cascade algorithm

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle( frame, (0,height-50), (200,height), (0,0,0), thickness=cv2.FILLED )

    # Rectangle surrounding face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1)

    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]                     # coordinates of right eye
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY) # converting color image to grayscale
        r_eye = cv2.resize(r_eye, (24,24))              # resizing image since our model's target size is (24, 24)
        r_eye = r_eye/255                               # scaling image vector between 0 and 1
        r_eye =  r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        rpred = model.predict(r_eye)                    # returns probabilities' score vector of length 2 [closed, open]
        rpred = np.argmax(rpred, axis=1)                # finding the index with largest score

        if(rpred[0] == 1):
            lbl='Open'
        if(rpred[0] == 0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]                     # coordinates of right eye
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY) # converting color image to grayscale
        l_eye = cv2.resize(l_eye, (24,24))              # resizing image since our model's target size is (24, 24)
        l_eye = l_eye/255                               # scaling image vector between 0 and 1
        l_eye =l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        lpred = model.predict(l_eye)                    # returns probabilities' score vector of length 2 [closed, open]
        lpred = np.argmax(lpred, axis=1)                # finding the index with largest score

        if(lpred[0]==1):
            lbl = 'Open'
        if(lpred[0]==0):
            lbl = 'Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score += 1
        cv2.putText(frame, lbl, (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

    else: # rpred[0]==1 or lpred[0]==1
        score -= 10
        cv2.putText(frame, lbl, (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)


    if(score<0):
        score = 0

    cv2.putText(frame,                  # frame
                'Score:' + str(score),  # text_to_be_displayed
                (100, height-20),        # coordinates
                font,                   # font_type
                1,                      # font_size : scaling factor relative to basic size of font
                (255,255,255),          # color
                1,                      # font_weight
                cv2.LINE_AA             # type_of_line_to_be_used (optional)
    )

    if(score>15):
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass

        if(thicc<16):
            thicc += 2
        else:
            thicc -= 2

            if(thicc<2):
                thicc = 2

        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thicc)

    # Display window
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
