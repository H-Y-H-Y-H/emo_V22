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
data_path = "../../EMO_GPTDEMO/robot_data/data1109/"

lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

select_lmks_id = lips_idx + inner_lips_idx
n_lmks = len(select_lmks_id)

key_cmds = np.asarray([0,1,2,3,5,7])

dataset_lmk = np.load(data_path+'m_lmks.npy')[:, select_lmks_id]
dataset_cmd = np.loadtxt(data_path+'action_tuned.csv')[:, key_cmds]

sample_id = np.arange(len(dataset_lmk))
np.random.shuffle(sample_id)
training_num = int(len(dataset_lmk) * 0.8)

tr_lmks = dataset_lmk[:training_num]
va_lmks = dataset_lmk[training_num:]
tr_cmds = dataset_cmd[:training_num]
va_cmds = dataset_cmd[training_num:]

init_cmds = np.asarray([9.999999999999997780e-02,
                        0.000000000000000000e+00,
                        5.555555555555555802e-01,
                        2.857142857142856984e-01,
                        5.384615384615384359e-01,
                        1.000000000000000000e+00
                        ])


class Robot_face_data(Dataset):
    def __init__(self, input_data, label_data, sequence=2,data_type_Flag = 0):
        self.input_data = input_data
        self.label_data = label_data
        self.input_data = torch.from_numpy(self.input_data).to(device, dtype=torch.float)
        self.label_data = torch.from_numpy(self.label_data).to(device, dtype=torch.float)
        self.init_cmds  = torch.from_numpy(init_cmds).to(device, dtype=torch.float)
        self.future_n_state = 2+2
        self.data_type_Flag = data_type_Flag
    def __getitem__(self, idx):

        if self.data_type_Flag == 0:
            cmds_0 = self.init_cmds
            cmds_1 = self.init_cmds

        elif self.data_type_Flag == 1:
            cmds_0 = self.init_cmds
            cmds_1 = self.label_data[idx+1]

        elif self.data_type_Flag == 2:
            cmds_0 = self.label_data[idx]
            cmds_1 = self.label_data[idx+1]
            noise_0 = torch.randn_like(cmds_0) * 0.1
            cmds_0 = cmds_0 + noise_0

            noise_1 = torch.randn_like(cmds_1) * 0.1
            cmds_1 = cmds_1 + noise_1

        else:
            cmds_0 = self.label_data[idx]
            cmds_1 = self.label_data[idx+1]

        rdm_future_state_id = idx + np.random.randint(2, self.future_n_state) # [low, high)
        # print('see the fucking rdm_future_state_id',rdm_future_state_id)

        lmks_2 = self.input_data[rdm_future_state_id]

        label_cmd = self.label_data[rdm_future_state_id]

        encoder_input = torch.cat((cmds_0.unsqueeze(0),cmds_1.unsqueeze(0)),dim=0)
        decoder_input = torch.transpose(lmks_2,1,0)


        sample = {"input": (encoder_input,decoder_input), "label": label_cmd}
        return sample

    def __len__(self):
        return len(self.input_data)-self.future_n_state

train_dataset = Robot_face_data(input_data=tr_lmks, label_data=tr_cmds)
test_dataset  = Robot_face_data(input_data=va_lmks, label_data=va_cmds,data_type_Flag=2)

# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)
# for i, bundle in enumerate(train_dataloader):
# input_d, label_d = bundle["input"], bundle["label"]


def train_model():
    wandb.init(project=project_name)
    config = wandb.config
    run_name = wandb.run.name

    print(run_name)

    log_path = "../data/%s/%s/"%(project_name,run_name)
    os.makedirs(log_path,exist_ok=True)

    # model = TransformerInverse(
    #             nhead              = config.nhead               ,
    #             num_encoder_layers = config.num_encoder_layers  ,
    #             num_decoder_layers = config.num_decoder_layers  ,
    #             dim_feedforward    = config.dim_feedforward
    #                       ).to(device)
    model = TransformerInverse().to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    Loss_fun = nn.L1Loss(reduction='mean')
    # Loss_fun = nn.L1Loss(reduction='mean')

    # You can use dynamic learning rate with this. Google it and try it!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)


    # mini test: check the shape of your data
    for mini_i in range(3):
        sample = train_dataset[mini_i]
        print(mini_i, sample['input'][0].shape,sample['input'][1].shape, sample['label'].shape)

    train_epoch_L = []
    test_epoch_L = []
    min_loss = + np.inf
    patience = 0


    for epoch in range(10000):
        t0 = time.time()
        model.train()
        temp_l = []
        running_loss = 0.0
        for data_type_id in range(1):
            train_dataloader.dataset.data_type_Flag = data_type_id%4
            for i, bundle in enumerate(train_dataloader):
                input_d, label_d = bundle["input"], bundle["label"]
                pred_result = model.forward(input_d[0],input_d[1])
                loss = Loss_fun(pred_result, label_d)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                temp_l.append(loss.item())
                running_loss += loss.item()

        train_mean_loss = np.mean(temp_l)
        train_epoch_L.append(train_mean_loss)

        model.eval()

        with torch.no_grad():

            # for data_type_id in range(3):
            #     train_dataloader.dataset.data_type_Flag = data_type_id % 3
            #     for i, bundle in enumerate(test_dataloader):
            #         input_d, label_d = bundle["input"], bundle["label"]
            #         pred_result = model.forward(input_d)
            #         # loss = model.loss(pred_result, label_d)
            #         loss = Loss_fun(pred_result, label_d)
            #         temp_l.append(loss.item())
            # outputs_data = [groundtruth_data[0], groundtruth_data[1]]
            temp_l = []
            pre_init_cmds = torch.from_numpy(init_cmds).to(device, dtype=torch.float).unsqueeze(0)  # Assuming init_cmds is a PyTorch tensor
            outputs       = torch.from_numpy(init_cmds).to(device, dtype=torch.float).unsqueeze(0)
            for i in range(len(test_lmk_data)):
                pre_pre_init_cmds = pre_init_cmds.clone()
                pre_init_cmds = outputs.clone()

                # Assuming test_lmk_data is a list of PyTorch tensors
                flatten_lmks = torch.transpose(test_lmk_data[i],1,0) # Flattening using PyTorch

                # Concatenation using PyTorch
                input_data = torch.cat((pre_pre_init_cmds, pre_init_cmds), dim=0)

                # Forward pass
                inputs_v = input_data.unsqueeze(0).to(device)
                flatten_lmks = flatten_lmks.unsqueeze(0).to(device)
                outputs = model.forward(inputs_v,flatten_lmks)

                # Loss calculation using PyTorch
                loss = torch.mean(torch.abs(outputs - groundtruth_data[i]))  # Assuming groundtruth_data is a tensor
                temp_l.append(loss.item())  # Convert tensor to a Python scalar


            test_mean_loss = np.mean(temp_l)
            test_epoch_L.append(test_mean_loss)


        scheduler.step(test_mean_loss)

        if min_loss>test_mean_loss:

            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(train_mean_loss))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(test_mean_loss))
            min_loss = test_mean_loss
            PATH = log_path + '/best_model_MSE.pt'
            torch.save(model.state_dict(), PATH)
            patience = 0
        patience += 1
        np.savetxt(log_path + "training_MSE.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_MSE.csv", np.asarray(test_epoch_L))

        t1 = time.time()
        wandb.log({"train_loss": train_mean_loss,
                   'valid_loss': test_mean_loss,
                   'epoch':epoch,
                   'learning_rate':optimizer.param_groups[0]['lr'],
                   'min_valid_loss': min_loss})

        print(epoch, "time used: ", round((t1 - t0),3),
              "training mean loss: ",round(train_mean_loss,5),
              "Test loss: ",test_mean_loss, "lr:", round(optimizer.param_groups[0]['lr'],5),
              " patience:",patience)
        if patience>20:
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
            'dim_feedforward':{"values":[128, 256, 512,1024]},
            'batchsize':{"values": [8, 16, 32,64]},
            'lr': {"max": 10e-5, "min":10e-6},
            'num_encoder_layers':{"values": [2,3,4,5,6]},
            'num_decoder_layers':{"values": [2,3,4,5,6]},
            'nhead':{"values": [1,2]}

        },
    }




    project_name = 'IVMT'
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)


    # output_dim = va_cmds.shape[1]
    # input_dim = n_lmks*3 + output_dim*2

    # print('input_dim:',input_dim)
    # print('output_dim:',output_dim)

    groundtruth_data = np.loadtxt(data_path + 'action_tuned.csv')[-training_num:, key_cmds]
    test_lmk_data = np.load(data_path+'m_lmks.npy')[-training_num:, select_lmks_id]

    groundtruth_data = torch.from_numpy(groundtruth_data).to(device, dtype=torch.float)
    test_lmk_data = torch.from_numpy(test_lmk_data).to(device, dtype=torch.float)

    # train_model()
    wandb.agent(sweep_id, function=train_model, count=100)
