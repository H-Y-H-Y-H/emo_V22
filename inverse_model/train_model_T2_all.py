from model import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
random.seed(0)

# Check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("start", device)
data_path = "../../../Downloads/data1201/"


lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

select_lmks_id = lips_idx + inner_lips_idx
n_lmks = len(select_lmks_id)

key_cmds = np.asarray([0,1,2,3,5,7])

dataset_lmk = np.load(data_path+'m_lmks.npy')[:, select_lmks_id]
dataset_cmd = np.loadtxt(data_path+'action.csv')[:, key_cmds]

training_num = int(len(dataset_lmk) * 0.8)

tr_lmks = dataset_lmk[:training_num]
va_lmks = dataset_lmk[training_num:]
tr_cmds = dataset_cmd[:training_num]
va_cmds = dataset_cmd[training_num:]

init_cmds = np.asarray([0.1,
                        0.0,
                        0.55556,
                        0.42857,
                        0.32353,
                        1.00000
                        ])

init_lmks = np.asarray(dataset_lmk[9])


class Robot_face_data(Dataset):
    def __init__(self, lmks_data, label_data, sequence=2, data_type_Flag=0):
        self.lmks_data = lmks_data
        self.label_data = label_data
        self.init_landmarks = torch.from_numpy(init_lmks).to(device, dtype=torch.float).flatten()
        self.lmks_data = torch.from_numpy(self.lmks_data).to(device, dtype=torch.float).flatten(1)
        self.label_data = torch.from_numpy(self.label_data).to(device, dtype=torch.float)
        self.init_cmds = torch.from_numpy(init_cmds).to(device, dtype=torch.float)
        self.future_n_state = 2 + 2
        self.data_type_Flag = data_type_Flag

    def __getitem__(self, idx):

        if self.data_type_Flag == 0:
            cmds_0 = self.init_cmds
            cmds_1 = self.init_cmds

            lmks_0 = self.lmks_data[idx]
            lmks_1 = self.lmks_data[idx + 1]

            cmds_label0 = self.label_data[idx]
            cmds_label1 = self.label_data[idx + 1]

        elif self.data_type_Flag == 1:
            cmds_0 = self.init_cmds
            cmds_1 = self.label_data[idx]

            lmks_0 = self.lmks_data[idx + 1]
            lmks_1 = self.lmks_data[idx + 2]

            cmds_label0 = self.label_data[idx + 1]
            cmds_label1 = self.label_data[idx + 2]

        elif self.data_type_Flag == 2:
            cmds_0 = self.label_data[idx]
            cmds_1 = self.label_data[idx + 1]

            lmks_0 = self.lmks_data[idx + 2]
            lmks_1 = self.lmks_data[idx + 3]

            cmds_label0 = self.label_data[idx + 2]
            cmds_label1 = self.label_data[idx + 3]

        # elif self.data_type_Flag == 3:
        #     cmds_0 = self.label_data[idx]
        #     cmds_1 = self.label_data[idx + 1]
        #
        #     lmks_0 = self.lmks_data[idx + 3]
        #     lmks_1 = self.lmks_data[idx + 4]
        #
        #     cmds_label0 = self.label_data[idx + 3]
        #     cmds_label1 = self.label_data[idx + 4]
        #
        # elif self.data_type_Flag == 4:
        #     cmds_0 = self.label_data[idx + 3]
        #     cmds_1 = self.label_data[idx + 2]
        #
        #     lmks_0 = self.lmks_data[idx + 1]
        #     lmks_1 = self.lmks_data[idx]
        #
        #     cmds_label0 = self.label_data[idx + 1]
        #     cmds_label1 = self.label_data[idx]
        #
        # else:
        #     cmds_0 = self.label_data[idx + 4]
        #     cmds_1 = self.label_data[idx + 3]
        #
        #     lmks_0 = self.lmks_data[idx + 1]
        #     lmks_1 = self.lmks_data[idx]
        #
        #     cmds_label0 = self.label_data[idx + 1]
        #     cmds_label1 = self.label_data[idx]

        encoder_input = torch.cat((cmds_0.unsqueeze(0), cmds_1.unsqueeze(0)), dim=0)
        target_lmks = torch.cat((lmks_0.unsqueeze(0), lmks_1.unsqueeze(0)), dim=0)
        label_cmds = torch.cat((cmds_label0.unsqueeze(0), cmds_label1.unsqueeze(0)), dim=0)

        sample = {"input_encoder": encoder_input, "input_decoder": target_lmks, "label_cmds": label_cmds}
        return sample

    def __len__(self):
        return len(self.lmks_data) - 4

train_dataset = Robot_face_data(lmks_data=tr_lmks, label_data=tr_cmds)
test_dataset  = Robot_face_data(lmks_data=va_lmks, label_data=va_cmds,data_type_Flag=2)

def train_model():
    wandb.init(project=project_name)
    # config = wandb.config
    run_name = wandb.run.name
    mode = 'all'
    print(run_name)

    log_path = "../data/%s/%s/"%(project_name,run_name)
    os.makedirs(log_path,exist_ok=True)

    model = TransformerInverse1207(decoder_input_size = 180,
                                nhead              = config.nhead               ,
                                num_encoder_layers = config.num_encoder_layers  ,
                                num_decoder_layers = config.num_decoder_layers  ,
                                dim_feedforward    = config.dim_feedforward,
                                   mode = mode
                          ).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    Loss_fun = nn.L1Loss(reduction='mean')
    # You can use dynamic learning rate with this. Google it and try it!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True)

    train_epoch_L = []
    test_epoch_L = []
    min_loss = + np.inf
    patience = 0

    relu = nn.ReLU()
    for epoch in range(10000):
        t0 = time.time()
        model.train()
        temp_l = []

        for data_type_id in range(5):
            train_dataloader.dataset.data_type_Flag = data_type_id%3
            for i, bundle in enumerate(train_dataloader):
                input_e,input_d, label_d = bundle["input_encoder"],bundle["input_decoder"], bundle["label_cmds"]
                pred_result = model.forward(input_e,input_d)
                loss = Loss_fun(pred_result, label_d)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                temp_l.append(loss.item())

        train_mean_loss = np.mean(temp_l)
        train_epoch_L.append(train_mean_loss)

        model.eval()
        with torch.no_grad():
            temp_l = []
            for data_type_id in range(5):
                train_dataloader.dataset.data_type_Flag = data_type_id % 3
                for i, bundle in enumerate(test_dataloader):
                    input_e, input_d, label_d = bundle["input_encoder"], bundle["input_decoder"], bundle["label_cmds"]
                    pred_result = model.forward(input_e, input_d)
                    loss = Loss_fun(pred_result, label_d)
                    temp_l.append(loss.item())
            valid_loss = np.mean(temp_l)

            test_epoch_L.append(valid_loss)

        scheduler.step(valid_loss)

        if min_loss>valid_loss:
            min_loss = valid_loss
            PATH = log_path + '/best_model_MSE.pt'
            torch.save(model.state_dict(), PATH)
            patience = 0
        patience += 1
        np.savetxt(log_path + "training_MSE.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_MSE.csv", np.asarray(test_epoch_L))

        t1 = time.time()
        wandb.log({'mode': mode,
                    "train_loss": train_mean_loss,
                   'valid_loss': valid_loss,
                   'epoch':epoch,
                   'learning_rate':optimizer.param_groups[0]['lr'],
                   'min_valid_loss': min_loss,
                   'dim_feedforward':config.dim_feedforward,
                   })

        print(epoch, "time used: ", round((t1 - t0),3),
              "training mean loss: ",round(train_mean_loss,5),
              "Test loss: ",round(valid_loss,5),
              "lr:", round(optimizer.param_groups[0]['lr'],5),
              " patience:",patience)
        if patience>45:
            break

    # plt.plot(np.arange(10,len(train_epoch_L)),train_epoch_L[10:])
    # plt.plot(np.arange(10,len(test_epoch_L)),test_epoch_L[10:])
    # plt.title('Learning Curve')
    # plt.legend()
    # plt.savefig(log_path+'lc.png')

if __name__ == '__main__':
    import wandb
    import argparse

    project_name = 'IVMT2_1207(encoder)'

    api = wandb.Api()
    pre_proj_name = 'IVMT2_1207(encoder)'
    run_id = 'eager-sweep-1' #'tough-grass-13'#'celestial-sweep-5'

    runs = api.runs("robotics/%s"%pre_proj_name)
    model_path = '../data/%s/%s/'%(pre_proj_name, run_id)
    config = None
    for run in runs:
        if run.name == run_id:
            print('loading configuration')
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    config = argparse.Namespace(**config)

    train_model()


