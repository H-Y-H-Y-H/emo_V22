# from servo_m import *
import cv2
import matplotlib.pyplot as plt
from realtime_landmark import *


if __name__ == "__main__":

    mode = 0

    ## Robot random landmarks dataset
    dataset_pth = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/data0914/'
    dataset_synthsize_pth = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/synthesized_lmks/'
    dataset_lmk = np.load(dataset_pth + 'm_lmks.npy')
    dataset_cmd = np.loadtxt(dataset_pth + 'action_tuned.csv')

    if mode == 0:
        data_path = 'data/desktop/'
        NORM_FLAG = False
        # Landmarks that the robot wants to mimic.
        target_lmks = np.load(data_path + 'emo_synced_lmks.npy')

        #####  SMOOTH LANDMARKS)
        # target_lmks = smooth_lmks(target_lmks)


        print(len(target_lmks))
        pts_size = 5
        logger_id = []
        distance_list = []
        current_action = 0
        mutli_nn_cmds_rank = 200
        if NORM_FLAG:
            img_savepath = "data/desktop/NN(BL)_dataset/norm_img(%dcmds_close)"%mutli_nn_cmds_rank
        else:
            img_savepath = "data/desktop/NN(BL)_dataset/img(%dcmds_close)"%mutli_nn_cmds_rank

        os.makedirs(img_savepath,exist_ok=True)
        for i in range(len(target_lmks)):

            # compute offline:
            lmks = target_lmks[i]

            min_dist_nn_id, rank_nn_id, rank_distance = nearest_neighber(
                lmks,
                dataset_lmk,
                only_mouth=True,
                normalize=NORM_FLAG,
                rank_data = mutli_nn_cmds_rank)
            five_action = dataset_cmd[rank_nn_id]
            dist = np.sum((five_action-current_action)**2,axis=1)
            min_id_action = np.argmin(dist)
            current_action = five_action[min_id_action]
            best_nn_id = rank_nn_id[min_id_action]
            print(i, best_nn_id,min_dist_nn_id)
            distance_list.append(rank_distance[min_id_action])

            # for sele_id in range(5):
            #     best_nn_id = rank5_nn_id[sele_id]
            #     print('FML')
            #     if best_nn_id < 2929 or best_nn_id > 3044:
            #         distance_list.append(five_distance[sele_id])
            #
            #         break

            # print(i,best_nn_id)

            fig, ax = plt.subplots(2,3)
            fig.suptitle('Normalized lmks Distance + cmds Distance(%d)'%mutli_nn_cmds_rank)

            fig.set_figheight(15)
            fig.set_figwidth(25)
            ax[0][0].scatter(lmks[:,0], -lmks[:,1], s=pts_size, label='Target')
            ax[0][0].scatter(dataset_lmk[best_nn_id,:, 0], -dataset_lmk[best_nn_id,:,1],s=pts_size, label='NN (dataset)')
            ax[0][1].scatter(-lmks[:,2],-lmks[:,1],s=pts_size, label='Target')
            ax[0][1].scatter(-dataset_lmk[best_nn_id,:,2], -dataset_lmk[best_nn_id,:,1],s=pts_size, label='NN (dataset)')
            ax[0][2].plot(distance_list)

            ax[0][0].legend()
            ax[0][1].legend()

            img_read_synthesized = plt.imread(dataset_synthsize_pth+'/%d.png'%i)#[:480,:640]
            img_read_dataset = plt.imread(dataset_pth+'img/%d.png'%best_nn_id)#[:480,:640]

            ax[1][0].imshow(img_read_synthesized)
            ax[1][0].title.set_text('Synthsized Image')

            ax[1][1].imshow(img_read_dataset)
            ax[1][1].title.set_text('Dataset Image')

            ax[1][2].imshow(img_read_dataset[:480,:640])
            ax[1][2].title.set_text('Output Image')

            # plt.
            ax[0][0].axis('equal')
            ax[0][1].axis('equal')
            # plt.show()
            plt.savefig(img_savepath+'/%d.jpeg'%i)
            plt.clf()
            plt.cla()
            plt.close()
            logger_id.append(best_nn_id)
        action_list = dataset_cmd[logger_id]

        if NORM_FLAG:
            np.savetxt('data/nvidia/emo_nn_id(rank%d)_norm.csv'%mutli_nn_cmds_rank, np.asarray(logger_id), fmt='%i')
            np.savetxt('data/nvidia/mimic_synced_cmds(rank%d)_norm.csv'%mutli_nn_cmds_rank, action_list)
        else:
            np.savetxt('data/nvidia/emo_nn_id(rank%d).csv' % mutli_nn_cmds_rank, np.asarray(logger_id), fmt='%i')
            np.savetxt('data/nvidia/mimic_synced_cmds(rank%d).csv' % mutli_nn_cmds_rank, action_list)

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


