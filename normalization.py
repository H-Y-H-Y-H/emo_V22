import numpy as np
import matplotlib.pyplot as plt


def find_H_static_face():
    # closest to robot static face.

    # Human landmarks data.
    input_data = np.load('data/en1_ava_lmks.npy')
    R_static_lmks = np.load('robo_norm/resting_m_lmks.npy')

    c_input_data = input_data.reshape(input_data.shape[0], -1)
    c_robotstatic_lmks = R_static_lmks.flatten()

    resting_face_lmks4compare = np.asarray([c_robotstatic_lmks] * len(c_input_data))
    #
    diff = np.mean(abs(resting_face_lmks4compare - c_input_data), axis=1)
    min_diff_idx = np.argmin(diff)
    print(min_diff_idx)
    plt.scatter(input_data[min_diff_idx, :, 0], input_data[min_diff_idx, :, 1])
    plt.show()

    # Just use id = 0.


def plot_two_face(face1, face2):
    plt.scatter(face1[0], face1[1])
    plt.scatter(face2[0], face2[1])
    plt.show()


def norm_array_landmarks(R_static_lmks, H_static_lmks, input_lmks, k=1):
    num_data = len(input_lmks)
    R_static_lmks_array = np.asarray([R_static_lmks] * num_data)
    H_static_lmks_array = np.asarray([H_static_lmks] * num_data)

    output_lmks = (input_lmks - H_static_lmks_array) * k + R_static_lmks_array

    return output_lmks


if __name__ == '__main__':
    input_data = np.load('data/en1_ava_lmks.npy')
    R_static_lmks = np.load('robo_norm/resting_m_lmks.npy')
    label_data = np.load('data/R_cmds_data.npy')
    resting_face_cmd = [0.0, 0.0, 0.8064516129032258, 0.8333333333333334, 0.42857142857142855,
                        0.4285714285714286, 0.875, 0.875, 1.0, 0.4285714285714286, 0.42857142857142855,
                        0.6666666666666666, 0.6666666666666667]

    H_static_id = 0
    H_static = input_data[H_static_id]

    output_d = norm_array_landmarks(R_static_lmks[0], H_static, input_data)

    np.save('data/en1_ava_lmks(norm).npy', output_d)
