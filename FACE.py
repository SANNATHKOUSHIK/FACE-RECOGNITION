import cv2
import pickle

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")

with open("labels.pickle",'rb') as file:
    labels = pickle.load(file)
    names = {N:n for n,N in labels.items()}



while True:
    ret, frame = cam.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors = 5)
    for (x,y,w,h) in face:
        region = gray[y:y+h,x:x+w]
        id_, confi = recognizer.predict(region)
        print(id_)
        print(names[id_])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255),1)
    cv2.imshow("hello", frame)
    if cv2.waitKey(20) == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
cam.release()