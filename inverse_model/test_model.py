import torch

from model import *
import os


def lmks2cmds(input_data, log_path):
    input_dim = input_data[0].shape[1] * input_data[0].shape[2]
    output_dim = 9

    model = inverse_model(input_size=input_dim, label_size=output_dim).to(device)
    model.load_state_dict(torch.load('../data/best_model_MSE.pt', map_location=torch.device(device)))
    model.eval()

    outputs_data = []
    for i in range(len(input_data)):
        inputs_v = torch.from_numpy(input_data[i].astype('float32')).to(device)
        inputs_v = torch.flatten(inputs_v, 1)

        outputs = model.forward(inputs_v)
        outputs = outputs.detach().cpu().numpy()
        outputs_data.append(outputs)

    outputs_log = np.concatenate(outputs_data, 0)
    np.savetxt(log_path, outputs_log)

    # loss = (outputs_log - label_data)**2
    # print('error:', np.mean(loss))


lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lmks_id = lips_idx + inner_lips_idx
    input_data = np.load('../data/en1_ava_lmks(norm).npy')[:, lmks_id]

    #
    # training_lmks = np.load('../data/R_lmks_data.npy')[0]
    # plt.scatter(training_lmks[:,0],training_lmks[:,1])
    # plt.scatter(input_data[14,:,0],input_data[14,:,1])
    # plt.show()

    chunksize = 64
    chunkperpare = [input_data[i:i + chunksize] for i in range(0, input_data.shape[0], chunksize)]

    lmks2cmds(chunkperpare, log_path="en_1_cmds.csv")
