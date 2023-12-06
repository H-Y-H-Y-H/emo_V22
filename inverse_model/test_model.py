import torch
from model import *
import os


def validation_dataset_test(dataset_lmk, groundtruth_data):
    outputs_data = [groundtruth_data[0], groundtruth_data[1]]
    for i in range(2, len(groundtruth_data)):
        print(i)
        pre_pre_init_cmds = groundtruth_data[i-2]
        pre_init_cmds = groundtruth_data[i-1]
        flatten_lmks = dataset_lmk[i].flatten()
        input_data = np.concatenate((pre_pre_init_cmds, pre_init_cmds, flatten_lmks))
        inputs_v = torch.from_numpy(input_data.astype('float32')).to(device)
        # inputs_v = torch.flatten(inputs_v, 1)
        inputs_v = inputs_v.unsqueeze(0)
        outputs = model.forward(inputs_v)
        outputs = outputs.detach().cpu().numpy()
        outputs_data.append(outputs[0])
        loss = np.mean((outputs[0] - groundtruth_data[i])**2)
        print(loss)

    loss = np.mean((np.asarray(outputs_data) - groundtruth_data)**2)
    print(loss)


def lmks2cmds(target_lmks, gt, log_path):
    # pre_init_cmds = groundtruth_data[0]
    # outputs = groundtruth_data[1]
    # outputs_data = [groundtruth_data[0],groundtruth_data[1]]

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

        outputs = model.forward(inputs_e,inputs_d)[0]
        outputs = outputs.detach().cpu().numpy()
        outputs_data.append(outputs)
        loss = np.mean(np.abs(outputs - gt[i]))
        # print(loss)
    final_loss = np.mean(np.abs(np.asarray(outputs_data) - gt[:-1]))
    print(final_loss)
    np.savetxt(log_path, outputs_data)
    return outputs_data


def use_model(target_lmks, log_path):
    # pre_init_cmds = groundtruth_data[0]
    # outputs = groundtruth_data[1]
    # outputs_data = [groundtruth_data[0],groundtruth_data[1]]

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
            flatten_lmks2 = target_lmks[i+1].flatten()

        input_data_e = np.concatenate((np.expand_dims(pre_pre_init_cmds,axis=0), np.expand_dims(pre_init_cmds,axis=0)))
        input_data_d = np.concatenate((np.expand_dims(flatten_lmks1,axis=0), np.expand_dims(flatten_lmks2,axis=0)))
        inputs_e = torch.from_numpy(input_data_e.astype('float32')).to(device).unsqueeze(0)
        inputs_d = torch.from_numpy(input_data_d.astype('float32')).to(device).unsqueeze(0)
        outputs = model.forward(inputs_e,inputs_d)[0]
        outputs = outputs.detach().cpu().numpy()
        outputs_data.append(outputs)
    np.savetxt(log_path, outputs_data)
    return outputs_data
    # loss = (outputs_log - label_data)**2
    # print('error:', np.mean(loss))

lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

if __name__ == '__main__':
    import wandb
    import argparse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/'

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
    proj_name = 'IVMT2_1205(closure)'
    run_id = 'wandering-sweep-1' #'tough-grass-13'#'celestial-sweep-5'
    pre_run_id = 'wandering-sweep-1'
    pre_proj_name = 'IVMT2_1205(closure)'


    runs = api.runs("robotics/%s"%pre_proj_name)

    model_config = run_id
    # model_name = "IVMT2_1202(finetuning)"
    # model_config = 'revived-wildflower-258' # 'laced-sweep-24'

    model_path = '../data/%s/%s/'%(proj_name, model_config)
    config = None
    for run in runs:
        if run.name == pre_run_id:
            print('loading configuration')
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    config = argparse.Namespace(**config)

    # model = inverse_model(input_size=input_dim,
    #                       label_size=output_dim,
    #                       num_layer=config.n_layer,
    #                       d_hidden=config.d_hidden,
    #                       use_bn=config.use_bn,
    #                       skip_layer=config.skip_layer,
    #                       final_sigmoid=config.final_sigmoid
    #                       ).to(device)
    model = TransformerInverse_baseline(decoder_input_size = 180,
                                nhead              = config.nhead               ,
                                num_encoder_layers = config.num_encoder_layers  ,
                                num_decoder_layers = config.num_decoder_layers  ,
                                dim_feedforward    = config.dim_feedforward,

                          ).to(device)
    # Load state dictionary
    state_dict = torch.load(model_path+'best_model_MSE.pt', map_location=torch.device(device))

    # Filter out unnecessary keys
    new_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    # Load the filtered state dictionary
    model.load_state_dict(new_state_dict, strict=False)

    # model.load_state_dict(torch.load(model_path+'best_model_MSE.pt', map_location=torch.device(device)))
    model.eval()

    mode = 0

    if mode == 0:
        data_path = "../../EMO_GPTDEMO/robot_data/data1201/"
        dataset_lmk = np.load(data_path + 'm_lmks.npy')
        mean_0 = np.mean(dataset_lmk[:, :1], axis=0)

        # use model to generate cmds
        save_path = f'../../EMO_GPTDEMO/robot_data/output_cmds/{run_id}/'
        os.makedirs(save_path, exist_ok=True)
        for demo_id in range(9,10):
            print(f'process: {demo_id}')
            target_lmks = np.load(d_root + f'robot_data/synthesized/lmks/m_lmks_{demo_id}.npy')[:, lmks_id]
            # dist = target_lmks[:,:1] - mean_0
            # target_lmks = target_lmks - dist

            use_model(target_lmks, log_path=save_path+f"{demo_id}.csv")

    elif mode == 1:
        # evaluations: lmks and cmds

        data_path = "../../EMO_GPTDEMO/robot_data/data1201/"
        dataset_lmk = np.load(data_path+'m_lmks.npy')
        groundtruth_data = np.loadtxt(data_path+'action.csv')

        mean_0 = np.mean(dataset_lmk[:, :1], axis=0)
        dist = dataset_lmk[:, :1] - mean_0
        dataset_lmk = dataset_lmk - dist

        training_num = int(len(dataset_lmk) * 0.8)

        key_cmds = np.asarray([0, 1, 2, 3, 5, 7])
        dataset_lmk = dataset_lmk[training_num:, lmks_id]
        groundtruth_data = groundtruth_data[training_num:, key_cmds]




        pred_cmds = lmks2cmds(dataset_lmk,groundtruth_data, log_path="../data/nvidia/pred_cmds.csv")

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


