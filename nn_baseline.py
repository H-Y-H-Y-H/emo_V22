# from servo_m import *
import cv2
import matplotlib.pyplot as plt
from realtime_landmark import *


if __name__ == "__main__":

    mode = 1


    ## Robot random landmarks dataset
    dataset_pth = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/data0914/'
    dataset_lmk = np.load(dataset_pth + 'm_lmks.npy')
    dataset_cmd = np.loadtxt(dataset_pth + 'action.csv')

    if mode == 0:
        data_path = 'data/desktop/'
 
        # Landmarks that the robot wants to mimic.
        target_lmks = np.load(data_path + 'emo_synced_lmks.npy')

        #####  SMOOTH LANDMARKS)
        target_lmks = smooth_lmks(target_lmks)


        print(len(target_lmks))

        logger_id = []
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
        np.savetxt('data/nvidia/smooth_emo_nn_id.csv', np.asarray(logger_id), fmt='%i')
        action_list = dataset_cmd[logger_id]
        np.savetxt('data/nvidia/smooth_mimic_synced_cmds.csv', action_list)

    elif mode == 1:

        trt_id = np.loadtxt('data/nvidia/emo_nn_id.csv', dtype=int)

        target_cmds = dataset_cmd[trt_id]
        print(target_cmds.shape)

        last_cmds = target_cmds[0]

        k = 0.6

        updated_id = []
        for i in range(len(target_cmds)):
            cur_cmds = target_cmds[i]
            updated_cmds = (cur_cmds-last_cmds)*k + last_cmds
            distances = np.linalg.norm(updated_cmds - dataset_cmd, axis=1)
            min_index = np.argmin(distances)

            print(min_index)
            updated_id.append(min_index)

        np.savetxt('data/nvidia/updated_emo_nn_id.csv',updated_id,fmt='%i')


