import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path= 'studentimages'
images=[]
classname=[]
mylist=os.listdir(path)
#print(mylist)
for cl in mylist:
    curimg =cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classname.append(os.path.splitext(cl)[0])
print(classname)
def findEncodings(images):
    encodeList = []
    for img in images:
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markattandance(name):
    with open('Attandance.csv','r+') as f:
          myDateList =f.readlines()
          nameList=[]
          for line in myDateList:
              entry =line.split(',')
              nameList.append(entry[0])
          if name not in nameList:
              now =datetime.now()
              dtString= now.strftime('%H:%M:%S')
              f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('encoding complete')

cap =cv2.VideoCapture(0)


while True:
    success,img =cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs= cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurentframe=face_recognition.face_locations(imgs)
    encodescurentframe= face_recognition.face_encodings(imgs,facesCurentframe)

    for encodeFace,faceLoc in zip(encodescurentframe,facesCurentframe):
        matches= face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis =face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name= classname[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 =faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattandance(name)
    cv2.imshow("web",img)
    cv2.waitKey(0)