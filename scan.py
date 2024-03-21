import cv2 as cv, time, os
from training import personFolders as pf
camera = 0
video = cv.VideoCapture(camera, cv.CAP_DSHOW)
haarCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read('training.xml')

# Check if video capture is successful
if not video.isOpened():
    print("Error: Could not open video capture.")
    exit()


while True:
    ret, frame = video.read()
    greyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = haarCascade.detectMultiScale(greyscale, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in face:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        ##
        id, conf = recognizer.predict(greyscale[y:y+h, x:x+w])
        
        ##
        #if conf <= 5:
        cv.putText(frame, pf[id-1], (x+10, y-10), cv.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
        #else:
        #    cv.putText(frame, 'unknown', (x+10, y-10), cv.FONT_HERSHEY_DUPLEX, 1, (0,255,0))


    cv.imshow("Face Recognition", frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv.destroyAllWindows()