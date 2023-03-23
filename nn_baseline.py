from servo_m import *
import cv2
import matplotlib.pyplot as plt

def nearest_neighber(lmks):


    duplicate_lmks = np.asarray([lmks]*len(dataset_lmk))
    distance = (duplicate_lmks - dataset_lmk)**2
    distance = np.mean(np.mean(distance,axis = 1),axis=1)
    rank = np.argsort(distance)
    best_nn_id = rank[0]


    best_nn_id_setid = (best_nn_id)//1000
    best_nn_id_imgid = best_nn_id%1000
    nn_img = cv2.imread(dataset_pth+add_dataset_pth[best_nn_id_setid]+'/img/%d.png'%best_nn_id_imgid)

    return best_nn_id, nn_img


if __name__ == "__main__":
    
    dataset_pth = '../'
    add_dataset_pth = ['dataset_resting1','dataset_lower0.5','dataset_pout0.5','dataset_upper0.5','dataset_smile0.5','dataset_resting0.5']

    dataset_lmk,dataset_cmd = [],[]

    for i in range(len(add_dataset_pth)):
        add_lmk = np.load(dataset_pth+add_dataset_pth[i]+'/m_lmks.npy')
        add_cmd = np.loadtxt(dataset_pth+add_dataset_pth[i]+'/action.csv')
        dataset_lmk.append(add_lmk)
        dataset_cmd.append(add_cmd)
    dataset_lmk, dataset_cmd = np.asarray(dataset_lmk), np.asarray(dataset_cmd)
    dataset_lmk = np.concatenate(dataset_lmk,0)[:,:468,:2]
    dataset_cmd = np.concatenate(dataset_cmd,0)
    print(dataset_lmk.shape)

    target_lmks = np.load('data/gpt_lmks/en-1.npy')


    step_n = 1

    mode = 1

    target_id = np.loadtxt('logger.csv')
    logger_id = []
    time_s = time.time()
    time0 = time.time()
    print(len(target_lmks))
    for i in range(0,len(target_lmks),step_n):
        print(i)
        if mode ==0:
            #compute offline:
            lmks = target_lmks[i].T
            lmks = lmks[:,:2]
            lmks = lmks/22 + 0.5
            lmks[:,1] = 1-lmks[:,1]
            lmks[:,1] -=0.05

            best_nn_id, nn_img = nearest_neighber(lmks)
            logger_id.append(best_nn_id)

        elif mode ==1:
            best_nn_id = int(target_id[i])
            cmds= dataset_cmd[best_nn_id]

            move_all(cmds,interval=1)
            time_left = time.time()-time0
            if 0.04*step_n-time_left>0:
                time.sleep(0.04*step_n-time_left)
            print(time_left)
            time0 = time.time()

    if mode == 0:
        np.savetxt('logger.csv',np.asarray(logger_id),fmt='%i')
        
    time_e = time.time()

    print(time_e - time_s)



        # select_lmk = dataset_lmk[best_nn_id]
        # plt.scatter(lmks[:,0],lmks[:,1],label='target')
        # plt.scatter(select_lmk[:,0],select_lmk[:,1],label='select')
        # plt.legend()

        # plt.show()
        # quit()

        # time.sleep(0.04)
