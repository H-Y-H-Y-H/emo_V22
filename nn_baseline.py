# from servo_m import *
import cv2
import matplotlib.pyplot as plt
from realtime_landmark import *


if __name__ == "__main__":

    mode = 0
    import sys
    d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/'
    ## Robot random landmarks dataset
    dataset_pth = d_root+'robot_data/data1109/'
    dataset_image_lmks = d_root+'robot_data/data1109/robot_dataset_img'
    dataset_synthsize_pth = d_root+'robot_data/data1109/data_lmks_media_syn'
    dataset_lmk = np.load(d_root+'robot_data/data1109/m_lmks.npy')
    dataset_cmd = np.loadtxt(dataset_pth + 'action.csv')
    print(dataset_lmk.shape)
    # all_images = []
    # for i in range(100):
    #     print(i)
    #     img = plt.imread(dataset_pth+'/img/%d.png'%i)[(480-320)//2:(480+320)//2:,(640-320)//2:(640+320)//2]
    #     all_images.append(img)
    # all_images = np.asarray(all_images)


    if mode == 0:
        NORM_FLAG = False
        mouth_re_localize = False
        # Landmarks that the robot wants to mimic.
        # target_lmks = np.load(data_path + 'emo_synced_lmks_close.npy')
        target_lmks = np.load(d_root+'robot_data/data1109/m_lmks_synthesized.npy')
        #####  SMOOTH LANDMARKS)
        # target_lmks = smooth_lmks(target_lmks)

        # dataset_lmk = dataset_lmk[:, mouth_lmks]

        print(len(target_lmks))

        logger_id = []
        distance_list = []
        current_action = 0
        mutli_nn_cmds_rank = 5

        # if NORM_FLAG:
        #     img_savepath = "data/desktop/NN(BL)_dataset/norm_img(%dcmds_close)"%mutli_nn_cmds_rank
        # else:
        img_savepath = d_root+"desktop/NN(BL)_dataset/img(%d_m_lmks_mouth)"%mutli_nn_cmds_rank

        os.makedirs(img_savepath,exist_ok=True)
        for i in range(len(target_lmks)):
            img_read_synthesized = plt.imread(dataset_synthsize_pth + '/%d.png' % i)  # [:480,:640]

            # compute offline:
            lmks = target_lmks[i]

            if mouth_re_localize:
                lmks = lmks[mouth_lmks]
                lmks -= lmks[0]
                dataset_lmk = dataset_lmk - dataset_lmk[:, :1]

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
            print(i, best_nn_id)
            distance_list.append(rank_distance[min_id_action])


            fig, ax = plt.subplots(2,3)
            fig.suptitle('Frame: %d. Normalized lmks Distance + cmds Distance(%d)'%(i,mutli_nn_cmds_rank))

            dataset_lmks_sele = dataset_lmk[best_nn_id]

            fig.set_figheight(15)
            fig.set_figwidth(25)
            # ax[0][0].scatter(lmks[0][0], -lmks[0][1], s=20, label='Target')
            # ax[0][0].scatter(lmks[mouth_lmks][:,0], -lmks[mouth_lmks][:,1], s=5, label='Target')
            # ax[0][0].scatter(dataset_lmks_sele[mouth_lmks][:,0], -dataset_lmks_sele[mouth_lmks][:,1],s=5, label='NN (dataset)')
            # ax[0][1].scatter(-lmks[mouth_lmks][:,2],-lmks[mouth_lmks][:,1],s=5, label='Target')
            # ax[0][1].scatter(-dataset_lmks_sele[mouth_lmks][:,2], -dataset_lmks_sele[mouth_lmks][:,1],s=5, label='NN (dataset)')
            # ax[0][2].plot(distance_list)

            # ax[0][0].scatter(lmks[0][0], -lmks[0][1], s=20, label='Target')
            ax[0][0].scatter(lmks[:,0], -lmks[:,1], s=5, label='Target')
            ax[0][0].scatter(dataset_lmks_sele[:,0], -dataset_lmks_sele[:,1],s=5, label='NN (dataset)')
            ax[0][1].scatter(-lmks[:,2],-lmks[:,1],s=5, label='Target')
            ax[0][1].scatter(-dataset_lmks_sele[:,2], -dataset_lmks_sele[:,1],s=5, label='NN (dataset)')
            ax[0][2].plot(distance_list)


            ax[0][0].legend()
            ax[0][1].legend()

            img_read_dataset = plt.imread(dataset_pth+'img/%d.png'%best_nn_id)#[:480,:640]
            image_lmks = plt.imread(dataset_image_lmks+'/%d.png'%best_nn_id)
            ax[1][0].imshow(img_read_synthesized)
            ax[1][0].title.set_text('Synthsized Image')

            ax[1][1].imshow(img_read_synthesized[:480,:480])
            ax[1][1].title.set_text('Synthsized Image')

            ax[1][2].imshow(img_read_dataset[:,(640-480)//2:(640+480)//2])
            ax[1][2].title.set_text('Dataset Image')


            ax[0][0].axis('equal')
            ax[0][1].axis('equal')
            # plt.show()
            plt.savefig(img_savepath+'/%d.jpeg'%i)
            plt.clf()
            plt.cla()
            plt.close()
            logger_id.append(best_nn_id)
        action_list = dataset_cmd[logger_id]

        # if NORM_FLAG:
        #     np.savetxt('data/nvidia/emo_nn_id(rank%d)_norm.csv'%mutli_nn_cmds_rank, np.asarray(logger_id), fmt='%i')
        #     np.savetxt('data/nvidia/mimic_synced_cmds(rank%d)_norm.csv'%mutli_nn_cmds_rank, action_list)
        # else:
        # np.savetxt('data/nvidia/emo_nn_id(rank%d).csv' % mutli_nn_cmds_rank, np.asarray(logger_id), fmt='%i')
        # np.savetxt('data/nvidia/mimic_synced_cmds(rank%d).csv' % mutli_nn_cmds_rank, action_list)
    # img_array = []
    # img_list = glob.glob('/Users/yuhan/PycharmProjects/EMO_GPTDEMO/data1105/img/*.png')
        img_array = []
        img_pth = 'data/desktop/NN(BL)_dataset/'
        img_list = os.listdir(img_savepath)
        print(len(img_list))
        for i in range(len(img_list)):
            filename = img_savepath+ "/%d.jpeg" % (i)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
            print(filename)

        out = cv2.VideoWriter(img_savepath + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30,size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

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


