# importing necessary packages numpy and open cv..
# create an environment with opencv installed and activate it before executing the file.
import numpy as np
import cv2
# just a simple haarcascade-based smile detection, but i will further update this in another file where i have trained a deep learning model on smile detection.

faceCascade = cv2.CascadeClassifier('models/detection/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('models/detection/haarcascade_smile.xml')
# capture the video stream with the camera
cap = cv2.VideoCapture(0)
# while the camera receives the live image
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #converting the BGR to GRAY scale to get rid of the channels
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces: # grabbing coordinates of the face in the image and width and height
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

        for i in smile:
            if len(smile) > 1:
                cv2.putText(img, "Smiling", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 3, cv2.LINE_AA)
     # showing the image
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
