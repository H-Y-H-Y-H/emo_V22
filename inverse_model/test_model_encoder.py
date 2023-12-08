import torch
from model import *
import os


class Robot_face_data(Dataset):
    def __init__(self, lmks_data, label_data, sequence=2,data_type_Flag = 0):
        self.lmks_data = lmks_data
        self.label_data = label_data
        self.init_landmarks = torch.from_numpy(init_lmks).to(device, dtype=torch.float).flatten()
        self.lmks_data = torch.from_numpy(self.lmks_data).to(device, dtype=torch.float).flatten(1)
        self.label_data = torch.from_numpy(self.label_data).to(device, dtype=torch.float)
        self.init_cmds  = torch.from_numpy(init_cmds).to(device, dtype=torch.float)
        self.future_n_state = 2+2
        self.data_type_Flag = data_type_Flag
    def __getitem__(self, idx):

        if self.data_type_Flag == 0:
            cmds_0 = self.init_cmds
            cmds_1 = self.init_cmds

            lmks_0 = self.init_landmarks
            lmks_1 = self.init_landmarks

        elif self.data_type_Flag == 1:
            cmds_0 = self.init_cmds
            cmds_1 = self.label_data[idx+1]

            lmks_0 = self.init_landmarks
            lmks_1 = self.lmks_data[idx+1]

        elif self.data_type_Flag == 2:
            cmds_0 = self.label_data[idx]
            cmds_1 = self.label_data[idx+1]

            lmks_0 = self.lmks_data[idx]
            lmks_1 = self.lmks_data[idx+1]

        elif self.data_type_Flag == 3:
            cmds_0 = self.label_data[idx+1]
            cmds_1 = self.label_data[idx]

            lmks_0 = self.lmks_data[idx+1]
            lmks_1 = self.lmks_data[idx]

        elif self.data_type_Flag == 4:
            cmds_0 = self.label_data[idx]
            cmds_1 = self.label_data[idx+2]

            lmks_0 = self.lmks_data[idx]
            lmks_1 = self.lmks_data[idx+2]

        else:
            cmds_0 = self.label_data[idx+2]
            cmds_1 = self.label_data[idx]

            lmks_0 = self.lmks_data[idx+2]
            lmks_1 = self.lmks_data[idx]

        encoder_input = torch.cat((cmds_0.unsqueeze(0),cmds_1.unsqueeze(0)),dim=0)
        label_lmks = torch.cat((lmks_0.unsqueeze(0),lmks_1.unsqueeze(0)),dim=0)


        sample = {"input": encoder_input, "label": label_lmks}
        return sample

    def __len__(self):
        return len(self.lmks_data)-2

def use_all_model(target_lmks, log_path):
    pre_init_cmds = init_cmds
    outputs = init_cmds
    outputs_data = []

    for i in range(len(target_lmks)):

        pre_pre_init_cmds = np.copy(pre_init_cmds)
        pre_init_cmds = np.copy(outputs)
        flatten_lmks1 = target_lmks[i].flatten()

        if i == len(target_lmks) - 1:
            flatten_lmks2 = target_lmks[i].flatten()
        else:
            flatten_lmks2 = target_lmks[i + 1].flatten()

        input_data_e = np.concatenate(
            (np.expand_dims(pre_pre_init_cmds, axis=0), np.expand_dims(pre_init_cmds, axis=0)))
        input_data_d = np.concatenate((np.expand_dims(flatten_lmks1, axis=0), np.expand_dims(flatten_lmks2, axis=0)))
        inputs_e = torch.from_numpy(input_data_e.astype('float32')).to(device).unsqueeze(0)
        inputs_d = torch.from_numpy(input_data_d.astype('float32')).to(device).unsqueeze(0)
        outputs = model.forward(inputs_e, inputs_d)[0][0]
        outputs = outputs.detach().cpu().numpy()
        outputs_data.append(outputs)
    np.savetxt(log_path, outputs_data)
    return outputs_data


def lmks_eval(target_lmks, gt):
    pre_init_cmds = init_cmds
    outputs = init_cmds
    outputs_data = []

    for i in range(len(target_lmks)-1):
        pre_pre_init_cmds = np.copy(pre_init_cmds)
        pre_init_cmds = np.copy(outputs)
        flatten_lmks1 = target_lmks[i].flatten()
        flatten_lmks2 = target_lmks[i+1].flatten()

        input_data_e = np.concatenate((np.expand_dims(pre_pre_init_cmds,axis=0), np.expand_dims(pre_init_cmds,axis=0)))
        input_data_d = np.concatenate((np.expand_dims(flatten_lmks1,axis=0), np.expand_dims(flatten_lmks2,axis=0)))

        inputs_e = torch.from_numpy(input_data_e.astype('float32')).to(device).unsqueeze(0)
        inputs_d = torch.from_numpy(input_data_d.astype('float32')).to(device).unsqueeze(0)

        outputs = model.forward(inputs_e,inputs_d)[0][0]
        outputs = outputs.detach().cpu().numpy()
        outputs_data.append(outputs)
        loss = np.mean(np.abs(outputs - gt[i]))
        print(loss)
    final_loss = np.mean(np.abs(np.asarray(outputs_data) - gt[:-1]))
    print(final_loss)

    return outputs_data





lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

if __name__ == '__main__':
    import wandb
    import argparse
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    lmks_id = lips_idx + inner_lips_idx

    init_cmds = np.asarray([0.1,
                            0.0,
                            0.55556,
                            0.42857,
                            0.32353,
                            1.00000
                            ])

    # MODEL LOADING
    api = wandb.Api()
    proj_name = 'IVMT2_1207(encoder)'
    run_id = 'eager-sweep-1' #'tough-grass-13'#'celestial-sweep-5'

    mode_all_id = 'proud-valley-19'

    runs = api.runs("robotics/%s"%proj_name)
    model_path = '../data/%s/%s/'%(proj_name, mode_all_id)
    config = None
    for run in runs:
        if run.name == run_id:
            print('loading configuration')
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    config = argparse.Namespace(**config)

    mode = 0
    if mode == 0:
        model = TransformerInverse1207(decoder_input_size=180,
                                       nhead=config.nhead,
                                       num_encoder_layers=config.num_encoder_layers,
                                       num_decoder_layers=config.num_decoder_layers,
                                       dim_feedforward=config.dim_feedforward,
                                       mode='all'
                                       ).to(device)

        model.load_state_dict(torch.load(model_path + 'best_model_MSE.pt', map_location=torch.device(device)))
        model.eval()

        d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/'
        data_path = "../../EMO_GPTDEMO/robot_data/data1201/"
        dataset_lmk = np.load(data_path + 'm_lmks.npy')
        init_lmks = dataset_lmk[9]

        # use model to generate cmds
        save_path = f'../../EMO_GPTDEMO/robot_data/output_cmds/{run_id}/'
        os.makedirs(save_path, exist_ok=True)

        use_model_or_eval_model = 1

        if use_model_or_eval_model == 0:
            for demo_id in range(8,11):
                print(f'process: {demo_id}')
                target_lmks = np.load(d_root + f'robot_data/synthesized/lmks/m_lmks_{demo_id}.npy')[:, lmks_id]
                use_all_model(target_lmks, log_path=save_path+f"{demo_id}.csv")

        else:
            groundtruth_data = np.loadtxt(data_path + 'action.csv')
            training_num = int(len(dataset_lmk) * 0.8)
            key_cmds = np.asarray([0, 1, 2, 3, 5, 7])
            dataset_lmk = dataset_lmk[training_num:, lmks_id]
            groundtruth_data = groundtruth_data[training_num:, key_cmds]
            lmks_eval(dataset_lmk,gt=groundtruth_data)

    if mode == 1:
        model = TransformerInverse1207(decoder_input_size=180,
                                       nhead=config.nhead,
                                       num_encoder_layers=config.num_encoder_layers,
                                       num_decoder_layers=config.num_decoder_layers,
                                       dim_feedforward=config.dim_feedforward,
                                       mode='encoder'
                                       ).to(device)

        model.load_state_dict(torch.load(model_path + 'best_model_MSE.pt', map_location=torch.device(device)))
        model.eval()


        # evaluations: lmks and cmds
        data_path = "../../../Downloads/data1201/"
        dataset_lmk = np.load(data_path+'m_lmks.npy')[:, lmks_id]
        input_cmds = np.loadtxt(data_path+'action.csv')
        init_lmks = np.asarray(dataset_lmk[9])
        training_num = int(len(dataset_lmk) * 0.8)

        key_cmds = np.asarray([0, 1, 2, 3, 5, 7])
        dataset_lmk = dataset_lmk[training_num:]
        input_cmds = input_cmds[training_num:, key_cmds]

        test_dataset = Robot_face_data(lmks_data=dataset_lmk, label_data=input_cmds, data_type_Flag=2)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
        Loss_fun = nn.L1Loss(reduction='mean')

        for i, bundle in enumerate(test_dataloader):
            test_dataset.data_type_Flag = 0
            input_d, label_d = bundle["input"], bundle["label"]
            pred_result = model.forward(input_d, 0)
            loss = Loss_fun(pred_result, label_d)

            pred_lmks = pred_result[0].detach().cpu().numpy().reshape(2,60,3)
            grdt_lmks = label_d[0].detach().cpu().numpy().reshape(2,60,3)

            for id_two in range(2):
                plt.scatter(pred_lmks[id_two,:,0],pred_lmks[id_two,:,1],label = 'pred')
                plt.scatter(grdt_lmks[id_two,:,0],grdt_lmks[id_two,:,1],label = 'gt')
                plt.legend()
                plt.show()
            print(loss)



    elif mode == 2:
        # use nn output as label to evaluate model:
        data_path = "../../EMO_GPTDEMO/robot_data/output_cmds/nn_5/"
        demo_id = 0

        for demo_id in range(10):
            dataset_lmk = np.load(data_path+f'lmks_{demo_id}.npy')


            mean_0 = np.mean(dataset_lmk[:, :1], axis=0)
            dist = dataset_lmk[:, :1] - mean_0
            dataset_lmk = dataset_lmk - dist

            groundtruth_data = np.loadtxt(data_path+f'cmds_{demo_id}.csv')

            key_cmds = np.asarray([0, 1, 2, 3, 5, 7])
            dataset_lmk = dataset_lmk[:,lmks_id]
            groundtruth_data = groundtruth_data[:, key_cmds]
            pred_cmds = lmks2cmds(dataset_lmk, groundtruth_data, log_path="pred_cmds.csv")


