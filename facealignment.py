# import face_alignment
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from skimage import io
# import collections
# import torch
#
# # Optionally set detector and some additional detector parameters
# face_detector = 'sfd'
# face_detector_kwargs = {
#     "filter_threshold" : 0.8
# }
#
# # Run the 3D face alignment on a test image, without CUDA.
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, dtype=torch.bfloat16, device='cuda', flip_input=True,
#                                   face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
# dataset_pth = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/data1105/'
#
# input_img = io.imread(dataset_pth+'img/0.png')
#
# preds = fa.get_landmarks(input_img)[-1]
#
# # 2D-Plot
# plot_style = dict(marker='o',
#                   markersize=4,
#                   linestyle='-',
#                   lw=2)
#
# pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
# pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
#               'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
#               'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
#               'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
#               'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
#               'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
#               'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
#               'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
#               'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
#               }
#
# fig = plt.figure(figsize=plt.figaspect(.5))
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(input_img)
#
# for pred_type in pred_types.values():
#     ax.plot(preds[pred_type.slice, 0],
#             preds[pred_type.slice, 1],
#             color=pred_type.color, **plot_style)
#
# ax.axis('off')
#
# # 3D-Plot
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.scatter(preds[:, 0] * 1.2,
#                   preds[:, 1],
#                   preds[:, 2],
#                   c='cyan',
#                   alpha=1.0,
#                   edgecolor='b')
#
# for pred_type in pred_types.values():
#     ax.plot3D(preds[pred_type.slice, 0] * 1.2,
#               preds[pred_type.slice, 1],
#               preds[pred_type.slice, 2], color='blue')
#
# ax.view_init(elev=90., azim=90.)
# ax.set_xlim(ax.get_xlim()[::-1])
# plt.show()
import cv2
import face_alignment
import torch
import numpy as np

data_path = 'C:/Users/yuhan/PycharmProjects/EMO_GPTDEMO/'

# Initialize the face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda')

lmks_data = []

def video_extraction():
    count = 0
    # Start video capture from the camera
    cap = cv2.VideoCapture(data_path+'synth_540.mp4')
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Predict landmarks
        frame = cv2.resize(frame,(480,480))

        preds = fa.get_landmarks(frame)
        if preds is not None:
            # preds is a list of arrays, where each array has shape (68, 2) for 2D or (68, 3) for 3D
            preds = preds[0]  # Assuming we take the landmarks of the first face detected
            # Draw landmarks on the frame
            for (x, y, z) in preds:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            lmks_data.append(preds)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        cv2.imwrite(data_path + 'robot_data/data_face_alig/%d.png'%count,frame)
        count += 1

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    np.save(data_path + 'fa_al_synthesized_lmks.npy', lmks_data)
    # When everything is done, release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

video_extraction()

def dataset_extraction():
    num_data = 46870
    data_path = 'C:/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/'

    for i in range(num_data):
        frame = cv2.imread(data_path+'data1105/img/%d.png'%i)
        frame = frame[:,80:560]
        preds = fa.get_landmarks(frame)
        if preds is not None:
            # preds is a list of arrays, where each array has shape (68, 2) for 2D or (68, 3) for 3D
            preds = preds[0]  # Assuming we take the landmarks of the first face detected
            # Draw landmarks on the frame
            for (x, y, z) in preds:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            lmks_data.append(preds)
        cv2.imwrite(data_path + '/data_lmks_faali/%d.png'%i,frame)
    np.save(data_path + 'fa_al_dataset_lmks.npy', lmks_data)

# dataset_extraction()