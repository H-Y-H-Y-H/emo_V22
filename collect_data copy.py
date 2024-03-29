from realtime_landmark import *
from hardware.servo_m import *
import queue, threading, time, os


# bufferless VideoCapture
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
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def render_img(image, face_mesh, pcf):
    image_original = np.copy(image)
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
            # rot_x = cv2.getTrackbarPos('vert', 'landmarks')
            # rot_y = cv2.getTrackbarPos('hori', 'landmarks')
            rot_x = 180
            rot_y = 270
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


if __name__ == "__main__":
    np.random.seed(2023)

    mode = 2

    # Collect robot babbling data:
    if mode == 0:
        # from servo_m import *
        save_data_pth = "../data1213/"
        os.makedirs(save_data_pth, exist_ok=True)
        os.makedirs(save_data_pth+'img/', exist_ok=True)

        cap = VideoCapture(4)

        # get cap property
        frame_width = cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
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

        # cv2.namedWindow("landmarks")
        # cv2.createTrackbar("vert", "landmarks", 180, 360, do_nothing)
        # cv2.createTrackbar("hori", "landmarks", 180, 360, do_nothing)
        img_i = 0
        r_lmks_logger = []
        m_lmks_logger = []
        action_logger = []

        step_num = 4
        NUM_data = 16*2000
        # TOTAL = Step * NUM_data
        num_motors = 8

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while 1:
                if img_i == 0:
                    for _ in range(100):
                        image = cap.read()
                    target_cmds = restart_face[:num_motors]
                else:
                    idx_ref = np.random.randint(0,17)
                    if idx_ref>14:
                        ref_face = init_cmds
                        target_cmds = np.random.normal(ref_face,scale=0.0)
                        
                    else:
                        ref_face = ref_list[idx_ref]
                        target_cmds = np.random.normal(ref_face,scale=0.1)

                    target_cmds = np.clip(target_cmds, 0, 1)
                    target_cmds = np.hstack((target_cmds[:4],target_cmds[3:4],target_cmds[4:5],target_cmds[4:5],target_cmds[5:6]))
                    
                # Get current motor joint angles:
                curr = np.zeros(num_motors)
                for i in range(num_motors):
                    curr[i] = all_motors[i].norm_v_cur

                # print("execute commands: ", target_cmds)
                # target_cmds[2], target_cmds[3] = check_lip_low(target_cmds[2], target_cmds[3])

                traj = np.linspace(curr, target_cmds, num=step_num, endpoint=True)

                # execute the commands:
                for i in range(step_num):
                    for j in range(num_motors):
                        val = traj[i][j]
                        all_motors[j].norm_act(val)
                    time.sleep(0.03)

                    image = cap.read()
                    # if not success:
                    #     print("Ignoring empty camera frame.")
                    #     # If loading a video, use 'break' instead of 'continue'.
                    #     continue

                    image_show, raw_lmks, m_lmks = render_img(image, face_mesh, pcf)
                    r_lmks_logger.append(raw_lmks)
                    m_lmks_logger.append(m_lmks)
                    action_logger.append(traj[i])

                    # SAVE
                    cv2.imwrite(save_data_pth+'img/%d.png' % img_i, image_show)
                    img_i += 1
                    if img_i % 20 == 0:
                        np.save(save_data_pth+'r_lmks.npy', np.asarray(r_lmks_logger))
                        np.save(save_data_pth+'m_lmks.npy', np.asarray(m_lmks_logger))
                        np.savetxt(save_data_pth+'action.csv', np.asarray(action_logger))
                        print(img_i, np.asarray(r_lmks_logger).shape)

                    # cv2.imshow('landmarks', image_show)

                if img_i >= NUM_data:
                    break
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.cap.release()

    # Collect landmarks from a video
    elif mode == 1:
        video_id = 9

        # save_path = f'../EMO_GPTDEMO/robot_data/synthesized/'
        # video_source = f'../EMO_GPTDEMO/robot_data/synthesized/video/{video_id}.mp4'

        real_data_method ='denim-dawn-82' # 'charmed-sky-46' #'true-sweep-2' #'nn_100' #'wav_bl'#'om'
        save_path = f'../EMO_GPTDEMO/robot_data/output_cmds/{real_data_method}_video/'
        video_source = f'../EMO_GPTDEMO/robot_data/output_cmds/{real_data_method}_video/{video_id}.mp4'

        os.makedirs(save_path + f'lmks_rendering/{video_id}/', exist_ok=True)
        os.makedirs(save_path + 'lmks/', exist_ok=True)

        print(video_source)
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # get cap property
        frame_width,frame_height = 480,480
        # frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        # frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
        cv2.createTrackbar("vert", "landmarks", 180, 360, do_nothing)
        cv2.createTrackbar("hori", "landmarks", 180, 360, do_nothing)
        img_i = 0
        r_lmks_logger = []
        m_lmks_logger = []
        action_logger = []

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            m_lmks = 0
            raw_lmks = 0
            image_show = 0

            count = 0
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # image = cv2.resize(image,(480,480))
                # image = image[(480-320)//2:(480+320)//2, (480-320)//2:(480+320)//2]
                print(image.shape)
                image_show, raw_lmks, m_lmks = render_img(image, face_mesh, pcf)
                print(image_show.shape)
                r_lmks_logger.append(raw_lmks)
                m_lmks_logger.append(m_lmks)

                cv2.imshow('landmarks', image_show)
                cv2.imwrite(save_path+'lmks_rendering/%d/%d.png'%(video_id,count),image_show)

                print(count)
                count+=1
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            np.save(save_path+'lmks/m_lmks_%d.npy'%video_id, m_lmks_logger)
            # np.save(save_path+'r_lmks_%d.npy'%video_id, r_lmks_logger)
        cap.release()

    elif mode == 2:

        img_source = "../EMO_GPTDEMO/robot_data/data1213/"
        os.makedirs(img_source+'robot_dataset_img',exist_ok=True)
        Num_data = 15000

        # get cap property
        frame_width = 480 #= cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        frame_height = 480#cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

        # cv2.namedWindow("landmarks")
        # cv2.createTrackbar("vert", "landmarks", 180, 360, do_nothing)
        # cv2.createTrackbar("hori", "landmarks", 180, 360, do_nothing)
        img_i = 0
        r_lmks_logger = []
        m_lmks_logger = []
        action_logger = []

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            m_lmks = 0
            raw_lmks = 0
            image_show = 0

            for i in range(Num_data):
                print(i)
                # ret, image = cap.read()
                # if not ret:
                #     print("Can't receive frame (stream end?). Exiting ...")
                #     break
                image = cv2.imread(img_source+'/img/%d.png'%i)
                # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                # image = np.dstack((image,image,image))
                image = image[:,(640-480)//2:(640+480)//2 ]

                image_show, raw_lmks, m_lmks = render_img(image, face_mesh, pcf)
                r_lmks_logger.append(raw_lmks)
                m_lmks_logger.append(m_lmks)

                # cv2.imshow('landmarks', image_show)
                cv2.imwrite(img_source+'/robot_dataset_img/%d.png'%i,image_show)

                # if cv2.waitKey(5) & 0xFF == 27:
                #     break

            np.save(img_source+'/m_lmks.npy', m_lmks_logger)
            np.save(img_source+'/r_lmks.npy', r_lmks_logger)

