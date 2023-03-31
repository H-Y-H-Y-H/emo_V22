import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import glob

def draw_lmks(lmks,label_lmk):
    plt.xlim(0, 1)
    plt.ylim(-1,0)
    plt.axis('equal')

    x = lmks[:,0]
    y = -lmks[:, 1]
    plt.scatter(x,y,s=5,label=label_lmk)
    plt.legend()

def two_lmks_compare():
    me_lmks = np.load('data/m_lmks_norm.npy')
    robo_lmks = np.load("/Users/yuhang/Downloads/en1_realrobot/m_lmks.npy")

    for i in range(len(robo_lmks)):
        draw_lmks(me_lmks[i],label_lmk='me')
        draw_lmks(robo_lmks[i],label_lmk='robo')
        plt.savefig('/Users/yuhang/Downloads/en1_realrobot/img2/%d.png'%i)
        plt.clf()

def combine_img():
    R_data = "/Users/yuhang/Downloads/en1_realrobot/img/"
    data2lmks = "/Users/yuhang/Downloads/en1_realrobot/img2/"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img_list = []
    for i in range(300):
        img_i = plt.imread(R_data+'%d.png'%i)
        img_i2 = plt.imread(data2lmks+'%d.png'%i)[...,:3]

        height, width, layers = img_i2.shape
        empty_img = np.zeros((height,width,layers))

        out_img = np.vstack((img_i2,empty_img))
        out_img = np.hstack((img_i,out_img))
        out_img = np.uint8(out_img*255)
        out_img = cv2.cvtColor(out_img,cv2.COLOR_RGB2BGR)
        img_list.append(out_img)

    width,height = img_list[0].shape[:2]
    img_size = (height,width)
    out = cv2.VideoWriter('/Users/yuhang/Downloads/en1_realrobot/project.mp4',fourcc, 30, img_size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()

combine_img()


