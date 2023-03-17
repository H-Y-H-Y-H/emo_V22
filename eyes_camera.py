import numpy as np
import cv2
import time
import mediapipe as mp
from servo_m import *
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

cap_r = cv2.VideoCapture(2)
cap_l = cv2.VideoCapture(5)

if not cap_r.isOpened():
    print("Cannot open right camera")
    exit()
if not cap_l.isOpened():
    print("Cannot open left camera")
    exit()



drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
landmarks_l,landmarks_r = 0.5,0.5
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

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

        frame_r.flags.writeable, frame_l.flags.writeable = False,False
        
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
        results_r = face_mesh.process(frame_r)
        results_l = face_mesh.process(frame_l)

        frame_r.flags.writeable, frame_l.flags.writeable = True, True
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)

        if results_r.multi_face_landmarks:
            for face_landmarks in results_r.multi_face_landmarks:
                landmarks_r = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

                mp_drawing.draw_landmarks(
                    image=frame_r,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame_r,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame_r,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        if results_l.multi_face_landmarks:
            for face_landmarks in results_l.multi_face_landmarks:
                landmarks_l = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

                mp_drawing.draw_landmarks(
                    image=frame_l,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame_l,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame_l,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        print(landmarks_l[6][0], landmarks_r[6][0])
        if results_l.multi_face_landmarks and results_r.multi_face_landmarks:
            eyes_move_2_traget(landmarks_l[6][0],landmarks_r[6][0])

        # Gray image
        # frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        # frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        
        combine_img = np.hstack((frame_l,frame_r))
        # Display the resulting frame
        cv2.imshow('frame', combine_img)
        if cv2.waitKey(1) == ord('q'):
            break
        t1 = time.time()
        print('fps:', round(1/(t1-t0),2))
# When everything done, release the capture
cap_r.release()
cap_l.release()
cv2.destroyAllWindows()
