import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import queue, threading, time

from face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def rotation_matrix(axis, theta):
    """
    Euler-Rodrigues formula
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def do_nothing(x):
    pass

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


def render_img(image_original):

    black_img = np.zeros(image_original.shape, dtype="uint8")
    black_img_rot = np.zeros(image_original.shape, dtype="uint8")
    black_img_metric = np.zeros(image_original.shape, dtype="uint8")

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image_original.flags.writeable = False
    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_original)

    # Draw the face mesh annotations on the image.
    image_original.flags.writeable = True
    image_original = cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            raw_lmks = np.copy(landmarks)
                
            # overlay landmarks on original image
            mp_drawing.draw_landmarks(
                image=image_original,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image_original,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image_original,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())


            # get metric landmarks
            landmarks_face_mesh = landmarks.copy()
            landmarks_face_mesh = landmarks_face_mesh.T[:, :468]

            metric_landmarks, pose_transform_mat_ = get_metric_landmarks(
                landmarks_face_mesh, pcf
            )

            # adjusting coordinates for better visualization
            metric_landmarks = metric_landmarks.T
            metric_landmarks = metric_landmarks / 22 + 0.5
            metric_landmarks[:, 1] = 1 - metric_landmarks[:, 1]
            metric_landmarks[:, 1] -= 0.05

            for idx, lm in enumerate(metric_landmarks):
                face_landmarks.landmark[idx].x = lm[0]
                face_landmarks.landmark[idx].y = lm[1]
                face_landmarks.landmark[idx].z = lm[2]

            # draw rotated landmarks
            mp_drawing.draw_landmarks(
                image=black_img_metric,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=black_img_metric,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())


            m_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])



            # freely rotate landmarks
            rot_x = cv2.getTrackbarPos('vert', 'landmarks')
            rot_y = cv2.getTrackbarPos('hori', 'landmarks')

            rot_y = 90
            rot_x_rad = (rot_x - 180) * np.pi / 180
            rot_y_rad = (rot_y - 180) * np.pi / 180

            rot_x_lm = m_landmarks @ rotation_matrix([1, 0, 0], rot_x_rad)
            rot_y_lm = rot_x_lm @ rotation_matrix([0, 1, 0], rot_y_rad)

            rot_y_lm[:, 0] -= rot_y_lm[:, 0].mean() - 0.5
            rot_y_lm[:, 1] -= rot_y_lm[:, 1].mean() - 0.5

            for idx, lm in enumerate(face_landmarks.landmark):
                lm.x = rot_y_lm[idx][0]
                lm.y = rot_y_lm[idx][1]
                lm.z = rot_y_lm[idx][2]

            # draw rotated landmarks
            mp_drawing.draw_landmarks(
                image=black_img_rot,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=black_img_rot,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())


    image_up = cv2.hconcat([image, image_original])
    image_dn = cv2.hconcat([black_img_metric, black_img_rot])

    image_show = cv2.vconcat([image_up, image_dn])

    return image_show, raw_lmks, m_landmarks


lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

def nearest_neighber(lmks):

    duplicate_lmks = np.asarray([lmks]*len(dataset))
    distance = (duplicate_lmks - dataset)**2
    distance = np.mean(np.mean(distance,axis = 1),axis=1)
    rank = np.argsort(distance)
    best_nn_id = rank[0]
    print(distance[rank[:5]])
    print(rank[:5])

    if best_nn_id>=1000:
        best_nn_id_setid = (best_nn_id-1000)//1000
        best_nn_id = best_nn_id%1000
        nn_img = cv2.imread(dataset_pth+add_dataset_pth[best_nn_id_setid]+'/img/%d.png'%best_nn_id)
    else:
        nn_img = cv2.imread(dataset_pth+'img/%d.png'%best_nn_id)
    return nn_img

if __name__ == "__main__":

    cap = VideoCapture(0)

    dataset_pth = '/Users/yuhang/Downloads/dataset1000_RF/'
    # dataset_pth = '../dataset/'
    dataset = np.load(dataset_pth+"m_lmks.npy")
    add_dataset_pth = ['dataset_lower0.5','dataset_pout0.5','dataset_upper0.5','dataset_smile0.5','dataset_resting0.5']
    for i in range(len(add_dataset_pth)):
        add_d = np.load(dataset_pth+add_dataset_pth[i]+'/m_lmks.npy')
        dataset = np.concatenate((dataset,add_d))


    # dataset = np.load(dataset_pth +"m_lmks.npy")


    # get cap property
    frame_width  = cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1]
    )

    cv2.namedWindow("landmarks")
    cv2.namedWindow("Robot")

    cv2.createTrackbar("vert", "landmarks", 180, 360, do_nothing)
    cv2.createTrackbar("hori", "landmarks", 180, 360, do_nothing)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while 1:
            image = cap.read()

            image_show, raw_lmks, m_lmks = render_img(image)

            nn_img = nearest_neighber(m_lmks)


            cv2.imshow('landmarks', image_show)
            cv2.imshow('Robot', nn_img)


            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
