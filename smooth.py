from visualization import *

def folder_frames2video(dir_pth):

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img_list = []

    num_frames = len(os.listdir(dir_pth))

    dir_pth2 = data_folder+'/lmks_smooth/smooth/'
    for i in range(num_frames):

        img_i = cv2.imread(dir_pth + '%d.png' % i)
        # img_i2 = cv2.imread(dir_pth2 + '%d.png' % i)
        # img_i = np.hstack((img_i,img_i2))
        img_list.append(img_i)

    width, height = img_list[0].shape[:2]
    img_size = (height, width)
    out = cv2.VideoWriter(dir_pth+'../%s.mp4'%lmks_type, fourcc, 30, img_size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()


if __name__ == "__main__":
    # partial_lmks_idx = lips_idx + inner_lips_idx
    data_folder = '/Users/yuhang/Downloads/EMO_GPTDEMO/'
    n_frames = 100
    lmks_type = 'smooth'

    lmks_list = np.load("data/R_lmks_data.npy")[:n_frames]

    # lmks_list = lmks_list[:,partial_lmks_idx]
    log_path = data_folder+'/lmks_smooth/%s/'%lmks_type
    print(log_path)
    os.makedirs(log_path, exist_ok=True)

    lmks_sm_list = smooth_lmks(lmks_list)

    for i in range(len(lmks_sm_list)):
        print(i)
        draw_lmks(lmks_sm_list[i],label_lmk='%s'%lmks_type)
        plt.savefig(log_path+'/%d.png' % i)
        plt.clf()

    dir_pth = data_folder+'/lmks_smooth/%s/'%lmks_type
    folder_frames2video(dir_pth)
