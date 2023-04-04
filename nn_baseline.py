# from servo_m import *
import cv2
import matplotlib.pyplot as plt
from realtime_landmark import *



if __name__ == "__main__":

    data_path = 'data/'
    dataset_pth = '/Users/yuhang/Downloads/dataset1000_RF/'
    add_dataset_pth = ['dataset_rdm_1_0', 'dataset_rdm_1_1', 'dataset_rdm_1_2', 'dataset_rdm_1_3',
                       'dataset_resting1', 'dataset_resting1(1)','dataset_resting1_10000']


    dataset_lmk = np.load(data_path + 'R_lmks_data.npy')
    dataset_cmd = np.load(data_path + 'R_cmds_data.npy')

    # target_lmks = np.load('data/gpt_lmks/en-1.npy')
    # target_lmks = np.load(data_path + 'm_lmks_norm.npy')
    target_lmks = np.load(data_path + 'en1_ava_lmks.npy')

    mode = 0

    # target_id = np.loadtxt('logger.csv')
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

            nn_img, best_nn_id = nearest_neighber(lmks, dataset_lmk[...,:3], add_dataset_pth, dataset_pth, only_mouth=True)

            # plt.scatter(lmks[:,0],lmks[:,1], label='me')
            # plt.scatter(dataset_lmk[best_nn_id,:,0],dataset_lmk[best_nn_id,:,1], label='nn_robot face')
            # plt.legend()
            # plt.show()

            logger_id.append(best_nn_id)
        np.savetxt('logger.csv', np.asarray(logger_id), fmt='%i')


    elif mode == 1:
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
