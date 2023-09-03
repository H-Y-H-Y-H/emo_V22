# from servo_m import *
import cv2
import matplotlib.pyplot as plt
from realtime_landmark import *

if __name__ == "__main__":

    data_path = 'data/'
    dataset_pth = '/Users/yuhang/Downloads/EMO_GPTDEMO/data0901/'

    dataset_lmk = np.load(dataset_pth + 'm_lmks.npy')
    dataset_cmd = np.loadtxt(dataset_pth + 'action.csv')

    # Landmarks that the robot wants to mimic.
    target_lmks = np.load(data_path + 'emo_synced_lmks.npy')

    #####  SMOOTH LANDMARKS)
    # target_lmks = smooth_lmks(target_lmks)
    # dataset_lmk = smooth_lmks(dataset_lmk)

    mode = 0

    logger_id = []
    time_s = time.time()
    time0 = time.time()
    print(len(target_lmks))

    frame_time = 1/30
    if mode == 0:
        for i in range(len(target_lmks)):
            print(i)
            # compute offline:
            lmks = target_lmks[i]
            lmks = lmks[:, :3]
            # lmks = lmks / 22 + 0.5
            # lmks[:, 1] = 1 - lmks[:, 1]
            # lmks[:, 1] -= 0.05

            nn_img, best_nn_id = nearest_neighber(lmks, dataset_lmk[..., :3], dataset_pth, only_mouth=True)

            # plt.scatter(lmks[:,0],lmks[:,1], label='me')
            # plt.scatter(dataset_lmk[best_nn_id,:,0],dataset_lmk[best_nn_id,:,1], label='nn_robot face')
            # plt.legend()
            # plt.show()

            logger_id.append(best_nn_id)
        np.savetxt('emo_purple_nn_id(smooth).csv', np.asarray(logger_id), fmt='%i')

    elif mode == 1:

        target_id = np.loadtxt('logger.csv')

        for i in range(len(target_lmks)):
            best_nn_id = int(target_id[i])
            cmds = dataset_cmd[best_nn_id]

            move_all(cmds, interval=1)
            time_left = time.time() - time0
            if frame_time - time_left > 0:
                time.sleep(frame_time - time_left)
            print(time_left)
            time0 = time.time()
        time_e = time.time()

        print(time_e - time_s)




    # select_lmk = dataset_lmk[best_nn_id]
    # plt.scatter(lmks[:,0],lmks[:,1],label='target')
    # plt.scatter(select_lmk[:,0],select_lmk[:,1],label='select')
    # plt.legend()

    # plt.show()
    # quit()

    # time.sleep(0.04)
