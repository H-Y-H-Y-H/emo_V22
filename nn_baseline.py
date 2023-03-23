from servo_m import *
import cv2
import matplotlib.pyplot as plt

def nearest_neighber(lmks):


    duplicate_lmks = np.asarray([lmks]*len(dataset_lmk))
    distance = (duplicate_lmks - dataset_lmk)**2
    distance = np.mean(np.mean(distance,axis = 1),axis=1)
    rank = np.argsort(distance)
    best_nn_id = rank[0]
    print(distance[rank[:5]])
    print(rank[:5])


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
    dataset_lmk = np.concatenate(dataset_lmk,0)[:,:468]
    dataset_cmd = np.concatenate(dataset_cmd,0)


    target_lmks = np.load('data/gpt_lmks/en-1.npy')

    logger_id = []
    for i in range(len(target_lmks)):
        lmks = target_lmks[i].T
        best_nn_id, nn_img = nearest_neighber(lmks)
        cmds= dataset_cmd[best_nn_id]
        logger_id.append(best_nn_id)

        select_lmk = dataset_lmk[best_nn_id]
        lmks = lmks/20 + 0.5
        # lmks[:,0] = lmks[:,0]+0.5
        # lmks[:,1] = lmks[:,1]+0.5
        lmks[:,1] = 1-lmks[:,1]
        plt.scatter(lmks[:,0],lmks[:,1],label='target')
        plt.scatter(select_lmk[:,0],select_lmk[:,1],label='select')
        plt.legend()

        plt.show()
        quit()


    # np.savetxt('logger.csv',np.asarray(logger_id),fmt='%i')
        # move_all(cmds)
        # time.sleep(0.04)
        



