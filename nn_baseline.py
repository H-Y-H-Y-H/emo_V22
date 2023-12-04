# from servo_m import *
import cv2
import matplotlib.pyplot as plt
from realtime_landmark import *


if __name__ == "__main__":

    mode = 0
    import sys
    demo_id = 1
    dataset_name = 'data1201'

    d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/'
    ## Robot random landmarks dataset
    dataset_pth = d_root+f'{dataset_name}/'
    dataset_image_lmks = d_root+f'{dataset_name}/robot_dataset_img'

    # dataset_pth_coarse= d_root+'data1126(coarse)/'
    # dataset_image_lmks_coarse = d_root+'data1126(coarse)/robot_dataset_img'

    target_synthesize_img_path = d_root + f'synthesized/lmks_rendering/{demo_id}'


    dataset_lmk = np.load(d_root+f'{dataset_name}/m_lmks.npy')
    dataset_cmd = np.loadtxt(d_root+f'{dataset_name}/action.csv')
    # dataset_lmk_coarse = np.load(d_root+'data1126(coarse)/m_lmks.npy')[:9810]
    # dataset_cmd_coarse = np.loadtxt(d_root+'data1126(coarse)/action.csv')[:9810]
    # dataset_lmk = np.concatenate((dataset_lmk,dataset_lmk_coarse))
    # dataset_cmd = np.concatenate((dataset_cmd,dataset_cmd_coarse))

    print(dataset_lmk.shape)

    if mode == 0:
        NORM_FLAG = False
        mouth_re_localize = True
        mutli_nn_cmds_rank = 200
        # Landmarks that the robot wants to mimic.
        # target_lmks = np.load(data_path + 'emo_synced_lmks_close.npy')
        target_lmks = np.load(d_root+f'synthesized/lmks/m_lmks_{demo_id}.npy')

        #####  SMOOTH LANDMARKS)
        # target_lmks = smooth_lmks(target_lmks)
        # dataset_lmk = dataset_lmk[:, mouth_lmks]

        print(len(target_lmks))

        logger_id = []
        distance_list = []
        current_action = 0
        nn_root = d_root + f'output_cmds/nn_{mutli_nn_cmds_rank}/'

        img_savepath = nn_root+ f'demo{demo_id}/visualization'
        os.makedirs(img_savepath, exist_ok=True)

        dataset_lmk = dataset_lmk - dataset_lmk[:, :1]

        for i in range(len(target_lmks)):
            img_read_synthesized = plt.imread(target_synthesize_img_path + '/%d.png' % i)  # [:480,:640]

            # compute offline:
            lmks = target_lmks[i]

            if mouth_re_localize:
                # lmks = lmks[mouth_lmks]
                lmks -= lmks[0]


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
            img_read_dataset = plt.imread(dataset_pth + 'img/%d.png' % best_nn_id)  # [:480,:640]
            image_lmks = plt.imread(dataset_image_lmks + '/%d.png' % best_nn_id)
            # if best_nn_id < 15260:
            #     img_read_dataset = plt.imread(dataset_pth+'img/%d.png'%best_nn_id)#[:480,:640]
            #     image_lmks = plt.imread(dataset_image_lmks+'/%d.png'%best_nn_id)
            # else:
            #
            #     img_read_dataset = plt.imread(dataset_pth_coarse+'img/%d.png'%(best_nn_id-15260))#[:480,:640]
            #     image_lmks = plt.imread(dataset_image_lmks_coarse+'/%d.png'%(best_nn_id-15260))
            ax[1][0].imshow(img_read_synthesized)
            ax[1][0].title.set_text('Synthsized Image')

            ax[1][1].imshow(img_read_synthesized[:480,:480])
            ax[1][1].title.set_text('Synthsized Image')

            ax[1][2].imshow(img_read_dataset[:,(640-480)//2:(640+480)//2])
            ax[1][2].title.set_text('Dataset Image')


            ax[0][0].axis('equal')
            ax[0][1].axis('equal')
            #plt.show()
            plt.savefig(img_savepath+'/%d.jpeg'%i)
            plt.clf()
            plt.cla()
            plt.close()
            logger_id.append(best_nn_id)
        action_list = dataset_cmd[logger_id]
        lmks_list =dataset_lmk[logger_id]

        np.savetxt(nn_root+f'nn_lmks_id_{demo_id}.csv', np.asarray(logger_id), fmt='%i')
        np.save(nn_root+f'lmks_{demo_id}.npy', lmks_list)
        np.savetxt(nn_root+f'cmds_{demo_id}.csv', action_list)

        img_array = []
        img_list = os.listdir(img_savepath)
        print(len(img_list))
        for i in range(len(img_list)):
            filename = img_savepath+ "/%d.jpeg" % (i)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
            print(filename)

        out = cv2.VideoWriter(nn_root + f'{demo_id}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30,size)

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


