import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import glob


def draw_lmks(lmks, label_lmk):
    plt.xlim(0, 1)
    plt.ylim(-1, 0)
    plt.axis('equal')

    x = lmks[:, 0]
    y = -lmks[:, 1]
    plt.scatter(x, y, s=5, label=label_lmk)
    plt.legend()


lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def two_lmks_compare():
    frame_id_list = np.loadtxt('emo_logger.csv')
    target_lmks0 = np.load('data/en1_emo_lmks.npy')
    target_lmks = np.load("/Users/yuhang/Downloads/EMO_GPTDEMO/real_emologger_noNorm_noFilter/en1_m_lmks.npy")
    robo_lmks = np.load("data/R_lmks_data.npy")
    log_pth = '/Users/yuhang/Downloads/EMO_GPTDEMO/real_emologger_noNorm_noFilter'
    os.makedirs(log_pth, exist_ok=True)
    for i in range(len(frame_id_list)):
        print(i)
        draw_lmks(target_lmks[i], label_lmk='real robot using synced video')
        draw_lmks(target_lmks0[i], label_lmk='synced video lmks')

        plt.legend()
        plt.savefig(log_pth+'/%d.png' % i)
        plt.clf()

        robo_lmks_id = frame_id_list[i]
        draw_lmks(robo_lmks[int(robo_lmks_id)], label_lmk='picked from robodata closest to synced video')
        draw_lmks(target_lmks0[i], label_lmk='synced video lmks')
        plt.legend()
        plt.savefig(log_pth+'/syncVSnn_%d.png' % i)
        plt.clf()



def combine_img():
    real_emo_pth = '/real_emologger_noNorm_F7_2/'
    dataPath = "/Users/yuhang/Downloads/EMO_GPTDEMO/"
    R_data = dataPath + real_emo_pth + "/img/"
    lmks_plot_pth = dataPath + real_emo_pth
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img_list = []
    length = len(np.load(dataPath+'en1_m_lmks.npy'))
    print(length)
    for i in range(length):
        img_i = plt.imread(R_data + '%d.png' % i)[:480]
        matplot_lmks = plt.imread(lmks_plot_pth + '%d.png' % i)[..., :3]
        # matplot_lmks2 = plt.imread(lmks_plot_pth + 'syncVSnn_%d.png' % i)[..., :3]

        height, width, layers = matplot_lmks.shape
        # empty_img = np.zeros((height, width, layers))

        # out_img = np.hstack((matplot_lmks, matplot_lmks2))
        out_img = np.hstack((img_i, matplot_lmks))
        out_img = np.uint8(out_img * 255)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        img_list.append(out_img)

    width, height = img_list[0].shape[:2]
    img_size = (height, width)
    out = cv2.VideoWriter('/Users/yuhang/Downloads/EMO_GPTDEMO/%s(robotandlmks).mp4'%real_emo_pth, fourcc, 30, img_size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()


# combine_img()


def image2video():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img_list = []
    dataset_pth = '/Users/yuhang/Downloads/dataset1000_RF/'
    add_dataset_pth = ['dataset_rdm_1_0', 'dataset_rdm_1_1', 'dataset_rdm_1_2', 'dataset_rdm_1_3',
                       'dataset_resting1', 'dataset_resting1(1)', 'dataset_resting1_10000']

    frame_id_list = np.loadtxt('logger(norm).csv')
    for i in range(len(frame_id_list)):
        best_nn_id = frame_id_list[i]
        if best_nn_id < 6000:
            best_nn_id_setid = best_nn_id // 1000
            best_nn_id = best_nn_id % 1000
        else:
            best_nn_id_setid = -1
            best_nn_id = best_nn_id - 6000

        nn_img = cv2.imread(dataset_pth + add_dataset_pth[int(best_nn_id_setid)] + '/img/%d.png' % best_nn_id)

        height, width, layers = nn_img.shape
        # empty_img = np.zeros((height,width,layers))
        #
        # out_img = np.vstack((img_i2,empty_img))
        # out_img = np.hstack((img_i,out_img))
        # out_img = np.uint8(out_img*255)
        # out_img = cv2.cvtColor(out_img,cv2.COLOR_RGB2BGR)
        img_list.append(nn_img)

    width, height = img_list[0].shape[:2]
    img_size = (height, width)
    out = cv2.VideoWriter('/Users/yuhang/Downloads/EMO_GPTDEMO/project(norm).mp4', fourcc, 30, img_size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()

# image2video()
