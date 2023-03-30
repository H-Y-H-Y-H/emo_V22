import torch

from model import *
import os

def lmks2cmds(input_data, log_path):


    model = inverse_model().to(device)
    model.load_state_dict(torch.load('../data/best_model_MSE.pt',map_location=torch.device(device)))
    model.eval()

    outputs_data = []
    for i in range(len(chunkperpare)):
        inputs_v = torch.from_numpy(chunkperpare[i].astype('float32')).to(device)
        inputs_v = inputs_v.reshape(inputs_v.shape[0],-1)
        outputs = model.forward(inputs_v)
        outputs = outputs.detach().cpu().numpy()
        outputs_data.append(outputs)

    np.savetxt(log_path,np.concatenate(outputs_data,0))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    input_data = np.load('../data/me_lmks_en1.npy')
    # input_data = input_data / 22 + 0.5
    # input_data[:,:, 1] = 1 - input_data[:,:, 1]
    # input_data[:,:, 1] -= 0.05

    # training_lmks = np.load('../data/R_lmks_data.npy')[0]
    # plt.scatter(training_lmks[:,0],training_lmks[:,2])
    # plt.scatter(input_data[0,:,0],input_data[0,:,2])
    # plt.show()



    chunksize = 64
    chunkperpare = [input_data[i:i+chunksize] for i in range(0,input_data.shape[0],chunksize)]

    lmks2cmds(chunkperpare, log_path = "logger_IVM(sota)/en_1.csv")


