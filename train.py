import cv2
import numpy
import os
import pickle
from PIL import Image


face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

x_train = []
label_ids = {}
y_labels = []
ids = 0

for root, dir, files in os.walk("images"):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root).replace(" ", "-").lower()
        # print(label, path)

        if not label in label_ids:
            label_ids[label] = ids
            ids += 1
        id_ = label_ids[label]
        # print(label_ids)

        image = Image.open(path).convert("L")
        img_array = numpy.array(image, 'uint8')
        # print(img_array)

        face = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors = 5)

        for x,y,w,h in face:
            region = img_array[y:y+h,x:x+w]
            x_train.append(region)
            y_labels.append(id_)

with open("labels.pickle", 'wb') as file:
    pickle.dump(label_ids, file)

recognizer.train(x_train, numpy.array(y_labels))
recognizer.save("train.yml")