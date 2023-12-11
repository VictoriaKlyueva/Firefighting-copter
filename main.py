from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
import sys

if __name__ == '__main__':
    path = os.path.join(os.path.abspath(os.curdir), 'saved_model.onnx')
    args_confidence = 0.2

    classes = ['fire', 'non fire']

    net = cv2.dnn.readNetFromONNX(path)

    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    frame = vs.read()
    frame = imutils.resize(frame, width=128*3, height=128*3)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), scalefactor=1.0 / 224
                                     , size=(224, 224), mean=(104, 117, 123), swapRB=True)

        net.setInput(blob)
        detections = net.forward()

        confidence = abs(detections[0][0] - detections[0][1])

        if (confidence > args_confidence):
            class_mark = np.argmax(detections)
            cv2.putText(frame, classes[class_mark], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 230, 220), 2)

        cv2.imshow("Web camera view", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        fps.update()

    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()
