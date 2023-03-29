from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def data_aug(eyes_data, mouth_data, eyes_cmds, mouth_cmds, seed=1, ts=0.2):
    print("Preparing data ...")
    # train test split
    train_input_eyes, test_input_eyes, train_label_eyes, test_label_eyes = train_test_split(eyes_data, eyes_cmds, test_size=ts, random_state=seed)
    train_input_mouth, test_input_mouth, train_label_mouth, test_label_mouth = train_test_split(mouth_data, mouth_cmds, test_size=ts, random_state=seed)

    # print(train_input_eyes.shape, train_input_mouth.shape, train_label_eyes.shape, train_label_mouth.shape)

    # train aug
    train_input = []
    train_label = []
    for i in range(len(train_input_eyes)):
        for j in range(len(train_input_mouth)):
            input_combine = np.concatenate((train_input_eyes[i], train_input_mouth[j]), axis=1)
            train_input.append(input_combine)

            label_combine = np.concatenate((train_label_eyes[i], train_label_mouth[j]), axis=0)
            train_label.append(label_combine)

    # test aug
    test_input = []
    test_label = []
    for i in range(len(test_input_eyes)):
        for j in range(len(test_input_mouth)):
            test_input_combine = np.concatenate((test_input_eyes[i], test_input_mouth[j]), axis=1)
            test_input.append(test_input_combine)

            test_label_combine = np.concatenate((test_label_eyes[i], test_label_mouth[j]), axis=0)
            test_label.append(test_label_combine)

    print("train: ", np.array(train_input).shape, np.array(train_label).shape)
    print("test: ", np.array(test_input).shape, np.array(test_label).shape)

    return np.array(train_input), np.array(train_label), np.array(test_input), np.array(test_label)


def minus_static(eyes_data, mouth_data, eyes_cmds, mouth_cmds):
    print("Preparing data (minus static face version) ...")
    static_lmks = np.concatenate((eyes_data[0], mouth_data[0]), axis=1)
    static_cmds = np.concatenate((eyes_cmds[0], mouth_cmds[0]), axis=0)

    eyes_data_new = eyes_data[1:]
    mouth_data_new = mouth_data[1:]
    eyes_cmds_new = eyes_cmds[1:]
    mouth_cmds_new = mouth_cmds[1:]

    train_input_eyes, test_input_eyes, train_label_eyes, test_label_eyes = train_test_split(eyes_data_new, eyes_cmds_new, test_size=0.2, random_state=1)
    train_input_mouth, test_input_mouth, train_label_mouth, test_label_mouth = train_test_split(mouth_data_new, mouth_cmds_new, test_size=0.2, random_state=2)

    # train aug
    train_input = []
    train_label = []
    for i in range(len(train_input_eyes)):
        for j in range(len(train_input_mouth)):
            input_combine = np.concatenate((train_input_eyes[i], train_input_mouth[j]), axis=1)
            input_minus = input_combine - static_lmks
            train_input.append(input_minus)

            label_combine = np.concatenate((train_label_eyes[i], train_label_mouth[j]), axis=0)
            label_minus = label_combine - static_cmds
            train_label.append(label_minus)

    # test aug
    test_input = []
    test_label = []
    for i in range(len(test_input_eyes)):
        for j in range(len(test_input_mouth)):
            test_input_combine = np.concatenate((test_input_eyes[i], test_input_mouth[j]), axis=1)
            test_input.append(test_input_combine - static_lmks)

            test_label_combine = np.concatenate((test_label_eyes[i], test_label_mouth[j]), axis=0)
            test_label.append(test_label_combine - static_cmds)

    print("train: ", np.array(train_input).shape, np.array(train_label).shape)
    print("test: ", np.array(test_input).shape, np.array(test_label).shape)

    return np.array(train_input), np.array(train_label), np.array(test_input), np.array(test_label)


if __name__ == "__main__":
    
    eyes_d = np.load("../../real_R_data/training_dataset/avg_eyes(609x2x26).npy")
    mouth_d = np.load("../../real_R_data/training_dataset/avg_mouth(1712x2x33).npy")
    eyes_c = np.load("../../real_R_data/training_dataset/eyes_cmds(609x4).npy")
    mouth_c = np.load("../../real_R_data/training_dataset/mouth_cmds(1712x7).npy")

    # data_aug(eyes_d, mouth_d, eyes_c, mouth_c)
    minus_static(eyes_d, mouth_d, eyes_c, mouth_c)