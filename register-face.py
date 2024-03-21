import cv2 as cv, os, numpy as np, shutil

camera = 0
video = cv.VideoCapture(camera, cv.CAP_DSHOW)
haarCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if video capture is successful
if not video.isOpened():
    print("Error: Could not open video capture.")
    exit()

name = input('Register Name : ')


# count dir inside path, check if exist, make new with number
dataPath = os.path.join(os.path.dirname(__file__), "data")
persons = os.listdir(dataPath)
# check if person exists
personNames = [person.split('.')[-1] for person in persons]

if name in personNames:
    index = personNames.index(name)
    # delete the old and replace with new one
    name = str(index+1) + '.' + name
    shutil.rmtree(dataPath + name)

else: 
    # otherwise create new by generating new index
    name = str(len(personNames)+1) + '.' + name
    

a = 0
while True:
    a += 1
    ret, frame = video.read()
    # convert image to grayscale for haarcascade model
    greyFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # greyFrame2 = np.array(frame, 'uint8')

    # detect face
    face = haarCascade.detectMultiScale(greyFrame, scaleFactor=1.3, minNeighbors=5)


    pathName = dataPath + '\\' + name 
    if not os.path.isdir(pathName):
        os.makedirs(pathName)

    for (x,y,w,h) in face:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv.imwrite(os.path.join(pathName, str(a) + '.jpg'), greyFrame[y:y+h, x:x+w])
    cv.imshow("Face Recognition", greyFrame)

    if (a > 29):
        break


video.release()
cv.destroyAllWindows()