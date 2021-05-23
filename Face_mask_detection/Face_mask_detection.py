import cv2 as cv
import keras.models as ml
import numpy
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

dataset=cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
model=ml.load_model("mask_recog_ver2.h5")
video=cv.VideoCapture(0)
preds=[]
while True:
    net,frame=video.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    grace=dataset.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(60,60),flags=cv.CASCADE_SCALE_IMAGE)
    for x,y,h,w in grace:
        frames=frame[y:y+h,w:w+h]
        gray=cv.cvtColor(frames,cv.COLOR_BGR2RGB)
        frames=cv.resize(gray,(224,224))
        array=numpy.array(frames)
        frames=numpy.expand_dims(array, axis=0)
        frames=preprocess_input(frames)
        faces_list=frames
        if len(faces_list) > 0:
            preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv.putText(frame, label, (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv.rectangle(frame,(x,y),(x+h,w+y),(255,0,0),2)
    cv.imshow("Mask Detection",frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()