import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
cap = cv.VideoCapture('en-1.mp4')
list_img = []
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # RGB_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow('frame', frame)
    list_img.append(frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


print(list_img)

