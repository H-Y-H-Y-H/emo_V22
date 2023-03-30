import numpy as np


def find_H_static_face():
    # closest to robot static face.

    # Human landmarks data.
    input_data = np.load('data/m_lmks.npy')

    lmks_data = np.load('data/R_lmks_data.npy')
    label_data = np.load('data/R_cmds_data.npy')
    resting_face_id = 5549

    c_input_data = input_data.reshape(input_data.shape[0],-1)
    c_robotstatic_lmks = lmks_data[resting_face_id].flatten()

    resting_face_lmks4compare = np.asarray([c_robotstatic_lmks]*len(c_input_data))
    #
    diff = np.mean(abs(resting_face_lmks4compare - c_input_data),axis=1)
    min_diff_idx = np.argmin(diff)
    print(min_diff_idx, label_data[min_diff_idx])

def plot_two_face(face1,face2):

    plt.scatter(face1[0],face1[1])
    plt.scatter(face2[0],face2[1])
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    input_data = np.load('data/m_lmks.npy')
    lmks_data = np.load('data/R_lmks_data.npy')
    label_data = np.load('data/R_cmds_data.npy')
    resting_face_cmd = [0.0, 0.0, 0.8064516129032258, 0.8333333333333334, 0.42857142857142855,
                        0.4285714285714286,0.525,0.525,1.0,0.4285714285714286,0.42857142857142855,0.6666666666666666,0.6666666666666667]
    R_static_id = 5549

    H_static_id = 14

    # plot_two_face((lmks_data[R_static_id][:,0], lmks_data[R_static_id][:,2]),
    #               (input_data[H_static_id][:,0],input_data[H_static_id][:,2]))

    H_static = input_data[H_static_id]
    R_static = lmks_data[R_static_id]
    input_data = (input_data - H_static) + R_static

    np.save('data/m_lmks_norm.npy',input_data)




