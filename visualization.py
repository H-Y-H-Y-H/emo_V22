import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import glob
from smooth import *
from scipy.signal import savgol_filter


def smooth_lmks(lmks_list,window = 7, order = 2):

    for j in range(len(lmks_list[0])):
        for i in range(2):
            lmks_list[:, j, i] = savgol_filter(lmks_list[:,j,i], window, order)
    return lmks_list


def draw_lmks(ax, lmks, label_lmk):
    # ax.xlim(0, 1)
    # ax.ylim(-1, 0)
    ax.axis('equal')

    x = lmks[:, 0]
    y = -lmks[:, 1]
    ax.scatter(x, y, s=5, label=label_lmk)



lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def two_lmks_compare():
    name = 'real_emo_gptdemo'
    frame_id_list = np.loadtxt('emo_purple_nn_id(smooth).csv')
    systh_lmks0 = np.load('data/en1_emo_lmks.npy')
    systh_lmks0 = smooth_lmks(systh_lmks0)

    real_lmks = np.load("/Users/yuhang/Downloads/EMO_GPTDEMO/new/%s/en1_m_lmks.npy"%name)

    # robo_lmks = np.load("data/R_lmks_data.npy")
    log_pth = '/Users/yuhang/Downloads/EMO_GPTDEMO/%s/'%name
    os.makedirs(log_pth, exist_ok=True)
    distance_list = []

    for i in range(len(frame_id_list)):
        print(i)

        distance = np.linalg.norm(real_lmks[i] - systh_lmks0[i])
        distance_list.append(distance)



    # distance_list = np.loadtxt(log_pth+'dist.csv')

    for i in range(len(frame_id_list)):
        print(i)

        # distance = np.linalg.norm(real_lmks[i] - systh_lmks0[i])
        # distance_list.append(distance)
        fig, ax = plt.subplots(2)
        fig.set_figheight(15)
        fig.set_figwidth(5)

        fig.suptitle('%s'%round(distance_list[i],4))

        draw_lmks(ax[0], real_lmks[i], label_lmk='real robot using synced video')
        draw_lmks(ax[0], systh_lmks0[i], label_lmk='synced video lmks')

        ax[1].plot(distance_list)
        ax[1].scatter( i, distance_list[i],c='r')

        ax[0].legend()
        plt.savefig(log_pth+'/%d.png' % i)
        plt.clf()

        # robo_lmks_id = frame_id_list[i]
        # draw_lmks(robo_lmks[int(robo_lmks_id)], label_lmk='picked from robodata closest to synced video')
        # draw_lmks(target_lmks0[i], label_lmk='synced video lmks')
        # plt.legend()
        # plt.savefig(log_pth+'/syncVSnn_%d.png' % i)
        # plt.clf()

    # plt.plot(np.arange(len(distance_list)), distance_list)
    # plt.show()

def combine_img():
    real_emo_pth = '/real_emo_sm/'
    dataPath = "/Users/yuhang/Downloads/EMO_GPTDEMO/new/real_emo_gptdemo/"
    R_data = dataPath + real_emo_pth + "/img/"
    lmks_plot_pth = "/Users/yuhang/Downloads/EMO_GPTDEMO/real_emo_gptdemo/"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img_list = []
    length = len(np.load(dataPath+'en1_m_lmks.npy'))
    print(length)
    for i in range(length):
        # img_i = plt.imread(R_data + '%d.png' % i)[:480]
        matplot_lmks = cv2.imread(lmks_plot_pth + '%d.png' % i)[..., :3]
        # matplot_lmks2 = plt.cv2(lmks_plot_pth + 'syncVSnn_%d.png' % i)[..., :3]
        # empty_img = np.zeros((height, width, layers))

        # out_img = np.hstack((matplot_lmks, matplot_lmks2))
        # out_img = np.hstack((img_i, matplot_lmks))
        # out_img = np.uint8(out_img * 255)
        # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        img_list.append(matplot_lmks)

    width, height = img_list[0].shape[:2]
    img_size = (height, width)
    out = cv2.VideoWriter('/Users/yuhang/Downloads/EMO_GPTDEMO/robotandlmks.mp4', fourcc, 30, img_size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()

def image2video():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img_list = []
    data_path = 'data/'
    dataset_pth = '/Users/yuhang/Downloads/EMO_GPTDEMO/data0901/'

    frame_id_list = np.loadtxt('emo_purple_nn_id(smooth).csv')
    for i in range(len(frame_id_list)):
        best_nn_id = frame_id_list[i]
        nn_img = cv2.imread(dataset_pth + '/img/%d.png' % best_nn_id)

        height, width, layers = nn_img.shape
        # empty_img = np.zeros((height,width,layers))

        # out_img = np.vstack((img_i2,empty_img))
        # out_img = np.hstack((img_i,out_img))
        # out_img = np.uint8(out_img*255)
        # out_img = cv2.cvtColor(out_img,cv2.COLOR_RGB2BGR)
        img_list.append(nn_img)

    width, height = img_list[0].shape[:2]
    img_size = (height, width)
    out = cv2.VideoWriter('/Users/yuhang/Downloads/EMO_GPTDEMO/project(sm).mp4', fourcc, 30, img_size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()

if __name__ == '__main__':
    # two_lmks_compare()

    # combine_img()

    image2video()