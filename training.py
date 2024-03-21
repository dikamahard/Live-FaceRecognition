import cv2, os
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer.create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

pathToData = os.path.join(os.path.dirname(__file__), "data")
personFolders = os.listdir(pathToData) # each person folder name

personPaths = [os.path.join(pathToData, person) for person in personFolders] # each person folder path


# modified imgPaths by copilot
imgPaths = [os.path.join(personPath, data) for personPath in personPaths for data in os.listdir(personPath)]


faceSamples = []
Ids = []

for imgPath in imgPaths:
    pilImage = Image.open(imgPath).convert('L')
    imgNp = np.array(pilImage, 'uint8')

    # get the person id from the path
    Id = int(imgPath.split('\\data\\')[-1].split('\\')[0].split('.')[0])

    faces = detector.detectMultiScale(imgNp)

    for(x,y,w,h) in faces:
        faceSamples.append(imgNp[y:y+h,x:x+w])
        Ids.append(Id)

recognizer.train(faceSamples, np.array(Ids))

# condition to update the training file
trainingPath = os.path.join(os.path.dirname(__file__), 'training.xml')

if os.path.exists(trainingPath):
    os.remove(trainingPath) # delete the old training
    recognizer.save('training.xml')
else:
    recognizer.save('training.xml')







'''

def getImagesWithLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)

        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples, Ids

'''

