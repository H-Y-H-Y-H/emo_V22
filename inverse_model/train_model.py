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
data_path = "../../EMO_GPTDEMO/data0914/"

lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

select_lmks_id = lips_idx + inner_lips_idx
n_lmks = len(select_lmks_id)

key_cmds = np.asarray([0,1,2,3,5,7])

dataset_lmk = np.load(data_path+'m_lmks.npy')[:, select_lmks_id]
dataset_cmd = np.loadtxt(data_path+'action_tuned.csv')[:, key_cmds]

dataset_lmk = dataset_lmk[3044:]
dataset_cmd = dataset_cmd[3044:]

sample_id = np.arange(len(dataset_lmk))
np.random.shuffle(sample_id)
training_num = int(len(dataset_lmk) * 0.8)

# tr_lmks = dataset_lmk[sample_id[:training_num]]
# va_lmks = dataset_lmk[sample_id[training_num:]]
# tr_cmds = dataset_cmd[sample_id[:training_num]]
# va_cmds = dataset_cmd[sample_id[training_num:]]
tr_lmks = dataset_lmk[:training_num]
va_lmks = dataset_lmk[training_num:]
tr_cmds = dataset_cmd[:training_num]
va_cmds = dataset_cmd[training_num:]



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


def train_model():
    wandb.init(project="IVM")
    config = wandb.config
    run_name = wandb.run.name


    print(run_name)

    log_path = "../data/logger_IVM/%s/"%run_name
    os.makedirs(log_path,exist_ok=True)


    model = inverse_model(input_size=input_dim,
                          label_size=output_dim,
                          num_layer=config.n_layer,
                          d_hidden=config.d_hidden,
                          use_bn=config.use_bn,
                          skip_layer=config.skip_layer,
                          final_sigmoid=config.final_sigmoid
                          ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    Loss_fun = nn.MSELoss(reduction='mean')
    # Loss_fun = nn.L1Loss(reduction='mean')

    # You can use dynamic learning rate with this. Google it and try it!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    train_dataset = Robot_face_data(input_data=tr_lmks, label_data=tr_cmds)
    test_dataset = Robot_face_data(input_data=va_lmks, label_data=va_cmds)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=True, num_workers=0)

    # mini test: check the shape of your data
    for mini_i in range(3):
        sample = train_dataset[mini_i]
        print(mini_i, sample['input'].shape, sample['label'].shape)


    train_epoch_L = []
    test_epoch_L = []
    min_loss = + np.inf
    patience = 0
    for epoch in range(10000):
        t0 = time.time()
        model.train()
        temp_l = []
        running_loss = 0.0
        for i, bundle in enumerate(train_dataloader):
            input_d, label_d = bundle["input"], bundle["label"]

            input_d = torch.flatten(input_d, 1)
            # label_d = torch.flatten(label_d, 1)

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
                # label_d = torch.flatten(label_d, 1)

                pred_result = model.forward(input_d)
                # loss = model.loss(pred_result, label_d)
                loss = Loss_fun(pred_result, label_d)
                temp_l.append(loss.item())

            test_mean_loss = np.mean(temp_l)
            test_epoch_L.append(test_mean_loss)
        scheduler.step(test_mean_loss)

        if min_loss-test_mean_loss>0.0001:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(train_mean_loss))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(test_mean_loss))
            min_loss = test_mean_loss
            PATH = log_path + '/best_model_MSE.pt'
            torch.save(model.state_dict(), PATH)
            patience = 0
        patience +=1
        np.savetxt(log_path + "training_MSE.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_MSE.csv", np.asarray(test_epoch_L))

        t1 = time.time()
        wandb.log({"train_loss": train_mean_loss,
                   'valid_loss': test_mean_loss,
                   'epoch':epoch,
                   'learning_rate':optimizer.param_groups[0]['lr']})

        print(epoch, "time used: ", round((t1 - t0),3), "training mean loss: ",round(train_mean_loss,5), "Test loss: ",test_mean_loss, "lr:", round(optimizer.param_groups[0]['lr'],5))
        if patience>30:
            break

    plt.plot(np.arange(10,len(train_epoch_L)),train_epoch_L[10:])
    plt.plot(np.arange(10,len(test_epoch_L)),test_epoch_L[10:])
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(log_path+'lc.png')




if __name__ == '__main__':
    import wandb
    # wandb.login()

    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "valid_loss"},
        "parameters": {
            'd_hidden':{"values":[128, 256, 512, 1024]},
            'batchsize':{"values": [8, 16, 32]},
            'learning_rate': {"max": 1e-6, "min":1e-8},
            'n_layer':{"values": [3, 4, 5]},
            'use_bn':{"values": [1,0]},
            'skip_layer':{"values": [1, 2]},
            'final_sigmoid': {"values": [1, 0]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="IVM")


    input_dim = n_lmks*3
    output_dim = va_cmds.shape[1]
    print('input_dim:',input_dim)
    print('output_dim:',output_dim)


    # run = wandb.init(project="IVM")


    # train_model()
    wandb.agent(sweep_id, function=train_model, count=100)
