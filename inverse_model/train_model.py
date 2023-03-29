from model import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
random.seed(0)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("start", device)
data_path = "/Users/yuhang/Downloads/dataset1000_RF/"
dataset_lmk = []
dataset_cmd = []
add_dataset_pth = ['dataset_rdm_1_0', 'dataset_rdm_1_1', 'dataset_rdm_1_2', 'dataset_rdm_1_3',
                   'dataset_resting1', 'dataset_resting1(1)', 'dataset_resting1_10000']

for i in range(len(add_dataset_pth)):
    add_d = np.load(data_path + add_dataset_pth[i] + '/m_lmks.npy')
    add_cmd = np.loadtxt(data_path + add_dataset_pth[i] + '/action.csv')

    dataset_lmk.append(add_d)
    dataset_cmd.append(add_cmd)
dataset_lmk = np.concatenate(dataset_lmk, axis=0)
dataset_cmd = np.concatenate(dataset_cmd, axis=0)
sample_id = random.sample(range(len(dataset_lmk)), len(dataset_lmk))

tr_lmks = dataset_lmk[sample_id[:int(len(dataset_lmk) * 0.8)]]
tr_cmds = dataset_cmd[sample_id[:int(len(dataset_cmd) * 0.8)]]

va_lmks = dataset_lmk[sample_id[int(len(dataset_lmk) * 0.8):]]
va_cmds = dataset_cmd[sample_id[int(len(dataset_cmd) * 0.8):]]


# INPUT SIZE:  2(dimensions) x 113(lmks)
# OUTPUT SIZE: 11(cmds)
class Robot_face_data(Dataset):
    def __init__(self, input_data, label_data):
        self.input_data = input_data
        self.label_data = label_data

    def __getitem__(self, idx):
        input_data_sample = self.input_data[idx]
        label_data_sample = self.label_data[idx]
        input_data_sample = torch.from_numpy(input_data_sample).to(device, dtype=torch.float)
        label_data_sample = torch.from_numpy(label_data_sample).to(device, dtype=torch.float)
        sample = {"input": input_data_sample, "label": label_data_sample}
        return sample

    def __len__(self):
        return len(self.input_data)


def train_model(batchsize, lr = 1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Loss_fun = nn.MSELoss(reduction='mean')
    # Loss_fun = nn.L1Loss(reduction='mean')

    # You can use dynamic learning rate with this. Google it and try it!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    train_dataset = Robot_face_data(input_data=tr_lmks, label_data=tr_cmds)
    test_dataset = Robot_face_data(input_data=va_lmks, label_data=va_cmds)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    # mini test: check the shape of your data
    for mini_i in range(3):
        sample = train_dataset[mini_i]
        print(mini_i, sample['input'].shape, sample['label'].shape)


    train_epoch_L = []
    test_epoch_L = []
    min_loss = + np.inf

    for epoch in range(num_epoches):
        t0 = time.time()
        model.train()
        temp_l = []
        running_loss = 0.0
        for i, bundle in enumerate(train_dataloader):
            input_d, label_d = bundle["input"], bundle["label"]

            input_d = torch.flatten(input_d, 1)
            label_d = torch.flatten(label_d, 1)

            pred_result = model.forward(input_d)
            loss = Loss_fun(pred_result, label_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_l.append(loss.item())
            running_loss += loss.item()

        train_mean_loss = np.mean(temp_l)
        train_epoch_L.append(train_mean_loss)

        model.eval()
        temp_l = []

        with torch.no_grad():
            for i, bundle in enumerate(test_dataloader):
                input_d, label_d = bundle["input"], bundle["label"]

                input_d = torch.flatten(input_d, 1)
                label_d = torch.flatten(label_d, 1)

                pred_result = model.forward(input_d)
                # loss = model.loss(pred_result, label_d)
                loss = Loss_fun(pred_result, label_d)
                temp_l.append(loss.item())

            test_mean_loss = np.mean(temp_l)
            test_epoch_L.append(test_mean_loss)
        scheduler.step(test_mean_loss)

        if test_mean_loss < min_loss:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(train_mean_loss))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(test_mean_loss))
            min_loss = test_mean_loss
            PATH = log_path + '/best_model_MSE.pt'
            torch.save(model.state_dict(), PATH)
        np.savetxt(log_path + "training_MSE.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_MSE.csv", np.asarray(test_epoch_L))

        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "training mean loss: ",train_mean_loss, "Test loss: ", test_mean_loss, "lr:", optimizer.param_groups[0]['lr'])

    plt.plot(np.arange(len(train_epoch_L)),train_epoch_L)
    plt.plot(np.arange(len(test_epoch_L)),test_epoch_L)
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(log_path+'lc.png')




if __name__ == '__main__':

    input_dim = va_lmks.shape[1]*va_lmks.shape[2]
    output_dim = va_cmds.shape[1]

    model = inverse_model(input_dim,output_dim).to(device)

    batchsize = 128  # 128
    num_epoches = 1000

    log_path = "../data/logger_IVM/"
    os.makedirs(log_path,exist_ok=True)

    train_model(batchsize)
