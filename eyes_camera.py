import numpy as np
import cv2 as cv
import time
cap_r = cv.VideoCapture(1)
cap_l = cv.VideoCapture(3)

if not cap_r.isOpened():
    print("Cannot open right camera")
    exit()
if not cap_l.isOpened():
    print("Cannot open left camera")
    exit()


while True:
    t0 = time.time()
    # Capture frame-by-frame
    ret_r, frame_r = cap_r.read()
    ret_l, frame_l = cap_l.read()
    # if frame is read correctly ret is True
    if not ret_r:
        print("Can't receive right eyes frame (stream end?). Exiting ...")
        break
    if not ret_l:
        print("Can't receive left eyes frame (stream end?). Exiting ...")
        break

    # Gray image
    # frame_r = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)
    # frame_l = cv.cvtColor(frame_l, cv.COLOR_BGR2GRAY)
    
    combine_img = np.hstack((frame_l,frame_r))
    
    # Display the resulting frame
    cv.imshow('frame', combine_img)
    if cv.waitKey(1) == ord('q'):
        break
    t1 = time.time()
    print(1/(t1-t0))
# When everything done, release the capture
cap_r.release()
cap_l.release()
cv.destroyAllWindows()