import cv2
import numpy as np
import face_recognition
imgelon= face_recognition.load_image_file('images/Elon Musk.jpg')
imgelon=cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgfasil= face_recognition.load_image_file('images/Faisal.jpg')
imgfasil=cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
faceLoc=face_recognition.face_locations(imgelon)[0]
encodeelon=face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLoc=face_recognition.face_locations(imgfasil)[0]
encodefasil=face_recognition.face_encodings(imgfasil)[0]
cv2.rectangle(imgfasil,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
cv2.imshow('elonmusk',imgelon)
cv2.imshow('fasil',imgfasil)
cv2.waitKey(0)