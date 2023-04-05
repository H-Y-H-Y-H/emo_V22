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

frame_id_list = np.loadtxt('logger.csv')


def two_lmks_compare():
    me_lmks = np.load('data/en1_ava_lmks.npy')
    # robo_lmks = np.load("/Users/yuhang/Downloads/EMO_GPTDEMO/m_lmks.npy")
    robo_lmks = np.load("data/R_lmks_data.npy")
    log_pth = '/Users/yuhang/Downloads/EMO_GPTDEMO/en1_ava_VS_NN'
    os.makedirs(log_pth,exist_ok=True)
    for i in range(len(me_lmks)):
        print(i)
        draw_lmks(me_lmks[i], label_lmk='me')

        robo_lmks_id = frame_id_list[i]

        draw_lmks(robo_lmks[int(robo_lmks_id)], label_lmk='nn')
        plt.legend()
        plt.savefig(log_pth+'/%d.png' % i)
        plt.clf()


# two_lmks_compare()


def combine_img():
    R_data = "/Users/yuhang/Downloads/EMO_GPTDEMO/img/"
    data2lmks = "/Users/yuhang/Downloads/EMO_GPTDEMO/en1_ava_VS_NN/"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img_list = []
    for i in range(300):
        img_i = plt.imread(R_data + '%d.png' % i)
        img_i2 = plt.imread(data2lmks + '%d.png' % i)[..., :3]

        height, width, layers = img_i2.shape
        empty_img = np.zeros((height, width, layers))

        out_img = np.vstack((img_i2, empty_img))
        out_img = np.hstack((img_i, out_img))
        out_img = np.uint8(out_img * 255)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        img_list.append(out_img)

    width, height = img_list[0].shape[:2]
    img_size = (height, width)
    out = cv2.VideoWriter('/Users/yuhang/Downloads/EMO_GPTDEMO/project_raw.mp4', fourcc, 30, img_size)

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

    frame_id_list = np.loadtxt('logger.csv')
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
    out = cv2.VideoWriter('/Users/yuhang/Downloads/EMO_GPTDEMO/project.mp4', fourcc, 30, img_size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()

image2video()
