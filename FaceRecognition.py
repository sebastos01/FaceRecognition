import numpy as np
from pyimagesearch import imutils
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('faces5.jpg')
ratio = img.shape[0] / 800.0
orig = img.copy()
img = imutils.resize(img, height = 800)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.rectangle(img,(15,0),(320,30),(0,0,0),-1)

cv2.putText(img,str(len(faces)) + " faces detected",(20,25),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
print str(len(faces))+" faces detected"


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()