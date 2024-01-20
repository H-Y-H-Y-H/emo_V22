from adafruit_servokit import ServoKit
import numpy as np
import random
import time
import board
import cv2
# i2c = board.I2C()
########################################
################# Face #################
########################################
pi = 3.14159
kit = [ServoKit(channels=16, address=0x41),
       ServoKit(channels=16, address=0x42)]


class Actuator(object):
    def __init__(self, idx_tuple, min_angle, max_angle, init_angle, inverse_flag=False):
        self.idx = idx_tuple  # data format: (kit number, channel)
        self.v_min = min_angle
        self.v_max = max_angle
        self.ranges = self.v_max - self.v_min
        self.v_cur = init_angle
        self.norm_v_cur = (self.v_cur - self.v_min) / self.ranges
        if inverse_flag == True:
            self.norm_v_cur = 1 - self.norm_v_cur
        self.norm_v_init = self.norm_v_cur
        self.inverse_value = inverse_flag
        self.norm_act(self.norm_v_init)

    # Input normalized cmds: [0,1]
    def norm_act(self, norm_cmd):
        if norm_cmd > 1.0 or norm_cmd < 0:
            print(norm_cmd, self.idx, "Norm cmd is not right, it should between 0 ~ 1.")
            quit()
        else:
            self.norm_v_cur = norm_cmd

            if self.inverse_value == True:
                norm_cmd = 1 - norm_cmd

            kit[self.idx[0]].servo[self.idx[1]].angle = norm_cmd * (self.ranges) + self.v_min

        self.v_cur = norm_cmd * (self.ranges) + self.v_min

    # Input angle: [0,180]
    def act(self, cmd):
        if cmd > self.v_max or cmd < self.v_min:
            print(cmd, "Cmd is out of action space")
            quit()
        else:
            kit[self.idx[0]].servo[self.idx[1]].angle = cmd
        self.v_cur = cmd
        self.norm_v_cur = (cmd - self.v_min) / self.ranges

    def ret(self):
        kit[self.idx[0]].servo[self.idx[1]].angle = self.v_min
        self.v_cur = self.v_min
        self.norm_v_cur = 0


def eyes_open():
    for i in range(100):
        print('eyeopen:',i)
        l_upper_eyelid.norm_act(np.sin((i + 1) / 200  * np.pi))
        r_upper_eyelid.norm_act(np.sin((i + 1) / 200  * np.pi))
        l_lower_eyelid.norm_act(np.cos((i + 1) / 200  * np.pi))
        r_lower_eyelid.norm_act(np.cos((i + 1) / 200  * np.pi))
        time.sleep(0.01)

# add value

lip_up = Actuator(idx_tuple=(0, 0), min_angle=70, max_angle=90, init_angle=88, inverse_flag=True)
lip_up_warp = Actuator(idx_tuple=(0, 1), min_angle=60, max_angle=140, init_angle=60)

# lip_down = Actuator(idx_tuple=(0, 4), min_angle=90, max_angle=125, init_angle=100)
# lip_down_warp = Actuator(idx_tuple=(0, 5), min_angle=90, max_angle=110, init_angle=90, inverse_flag=True)
#  lip_down_warp_cmds = np.random.uniform((0.8-lip_down_cmds),1)
lip_down = Actuator(idx_tuple=(1, 11), min_angle=80, max_angle=125, init_angle=100, inverse_flag=True)
# lip_down.norm_act(0)


r_corner_up = Actuator(idx_tuple=(0, 2), min_angle=40, max_angle=110, init_angle=70)
l_corner_up = Actuator(idx_tuple=(0, 7), min_angle=20, max_angle=90, init_angle=55, inverse_flag=1)

r_corner_low = Actuator(idx_tuple=(0, 3), min_angle=30, max_angle=98, init_angle=52)
l_corner_low = Actuator(idx_tuple=(0, 6), min_angle=85, max_angle=155, init_angle=138, inverse_flag=1)

r_eye_yaw = Actuator(idx_tuple=(0, 8), min_angle=65, max_angle=125, init_angle=95)
l_eye_yaw = Actuator(idx_tuple=(1, 7), min_angle=60, max_angle=120, init_angle=90)

r_eye_pitch = Actuator(idx_tuple=(0, 9), min_angle=68, max_angle=98, init_angle=83)
l_eye_pitch = Actuator(idx_tuple=(1, 6), min_angle=78, max_angle=108, init_angle=93, inverse_flag=True)

# lip_up.norm_act(0.4)
# lip_up_warp.norm_act(1)
# l_corner_up.norm_act(0.4)
# r_corner_up.norm_act(0.4)
# l_corner_low.norm_act(1)
# r_corner_low.norm_act(1)
# lip_down_warp.norm_act(0)
# lip_down.norm_act(0.8)


start_as_closed_eyes = False
if start_as_closed_eyes:
    r_upper_eyelid = Actuator(idx_tuple=(0, 11), min_angle=80, max_angle=153, init_angle=153, inverse_flag=True)
    l_upper_eyelid = Actuator(idx_tuple=(1, 4), min_angle=28, max_angle=100, init_angle=28)
    r_lower_eyelid = Actuator(idx_tuple=(0, 10), min_angle=93, max_angle=143, init_angle=143)
    l_lower_eyelid = Actuator(idx_tuple=(1, 5), min_angle=40, max_angle=95, init_angle=40, inverse_flag=True)
    time.sleep(1)
    eyes_open()

else:
    r_upper_eyelid = Actuator(idx_tuple=(0, 11), min_angle=80, max_angle=153, init_angle=80, inverse_flag=True)
    l_upper_eyelid = Actuator(idx_tuple=(1, 4), min_angle=28, max_angle=100, init_angle=100)

    r_lower_eyelid = Actuator(idx_tuple=(0, 10), min_angle=93, max_angle=143, init_angle=93)
    l_lower_eyelid = Actuator(idx_tuple=(1, 5), min_angle=40, max_angle=95, init_angle=95, inverse_flag=True)


r_inner_eyebrow = Actuator(idx_tuple=(0, 14), min_angle=65, max_angle=100, init_angle=85, inverse_flag=True)
l_inner_eyebrow = Actuator(idx_tuple=(1, 1), min_angle=75, max_angle=110, init_angle=90)

r_outer_eyebrow = Actuator(idx_tuple=(0, 15), min_angle=80, max_angle=125, init_angle=110)
l_outer_eyebrow = Actuator(idx_tuple=(1, 0), min_angle=60, max_angle=105, init_angle=75, inverse_flag=True)

jaw = Actuator(idx_tuple=(1, 15), min_angle=40, max_angle=94, init_angle=94)

neck_mode = False
neck_yaw = Actuator(idx_tuple=(1, 14), min_angle=20, max_angle=160, init_angle=85)

if neck_mode == True:

    neck_roll = Actuator(idx_tuple=(1, 13), min_angle=90, max_angle=120, init_angle=105)
    neck_pitch = Actuator(idx_tuple=(1, 12), min_angle=20, max_angle=60, init_angle=40, inverse_flag=1)
    all_motors = [lip_up, lip_up_warp, lip_down, r_corner_up, l_corner_up, r_corner_low, l_corner_low,
                  jaw,
                  r_inner_eyebrow, l_inner_eyebrow, r_outer_eyebrow, l_outer_eyebrow,
                  neck_roll, neck_pitch, neck_yaw]
else:
    all_motors = [lip_up, lip_up_warp, lip_down,
                  r_corner_up, l_corner_up, r_corner_low, l_corner_low,
                  jaw,
                  r_inner_eyebrow, l_inner_eyebrow, r_outer_eyebrow, l_outer_eyebrow]

# def check_lip_low(cmd_lip_down, cmd_lip_down_warp):
#     # regenerate the lip values.
#     if cmd_lip_down + cmd_lip_down_warp < 0.8:
#         cmd_lip_down_warp = np.random.uniform((0.8 - cmd_lip_down), 1)
#         print('reproduce cmd_lip_down_warp ', )

    # return cmd_lip_down, cmd_lip_down_warp



def check_lip_upper(cmd_lip_up,cmd_lip_up_warp,cmd_lip_low,cmd_corner_up,cmd_corner_low):

    if (cmd_corner_low<0.3) and (cmd_corner_up<0.3) and (cmd_lip_up_warp>0.5):
        print('abnormality 0')
        cmd_lip_up_warp = random.uniform(0.,0.5)
    
    # # Abnormality 1:
    # if (cmd_corner_up>0.7) and (cmd_lip_up<0.5):
    #     print('abnormality 1')
    #     cmd_lip_up = random.uniform(0.5,1)
    
    # # Abnormality 2:
    # if (cmd_corner_up<0.3) and (cmd_corner_low>0.8) and (cmd_lip_up_warp>0.8):
    #     print('abnormality 2')
    #     cmd_lip_up = random.uniform(0,0.5)

    # Abnormality 2:
    if (cmd_corner_up<0.2) and (cmd_corner_low>0.8):
        print('abnormality 2')
        cmd_corner_up = random.uniform(0.6,1)
        # cmd_lip_up_warp = random.uniform(0.8,1)
    
    # Abnormality 3:
    if (cmd_lip_up > 0.6) and (cmd_lip_up_warp > 0.4): 
        print('abnormality 3')
        cmd_lip_up_warp = random.uniform(0,0.4)
    
    # Abnormality 4:
    if (cmd_lip_up_warp>0.8) and (cmd_lip_up>0.3):
        print('abnormality 4')
        cmd_lip_up_warp = random.uniform(0,0.8)


    # if (cmd_corner_up<0.3):
    #     print('abnormality 4')
    #     cmd_lip_up_warp = random.uniform(0,0.3)

    # # Abnormality 5:
    # if (cmd_corner_up<0.3 ) and (cmd_corner_low>0.8):
    #     print('abnormality 5')
    #     cmd_lip_up = random.uniform(0.9,1)
    #     cmd_lip_up_warp = random.uniform(0.4,0.5)
    #     cmd_lip_low = random.uniform(0.3,1)


    # #Abnormality 5:
    # if cmd_lip_low<0.4:


    return cmd_lip_up, cmd_lip_up_warp, cmd_lip_low, cmd_corner_up


def random_cmds(reference=None, noise=0.2, only_mouth=True):
    num_motors = len(all_motors)

    # if reference != None:
    cmds_random = np.random.normal(reference, scale=noise)
    # else:
    #     # Generate random movement and check lower lips
    #     cmds_random = np.random.sample(num_motors)
    cmds_random = np.clip(cmds_random, 0, 1)

    # cmds_random = [1. , 1 ,0.68233284, 0.53335575, 0.53335575, 0.0, 0.0, 0.37913007, 0.42857, 0.42857, 0.66667, 0.66667]

    # Symmetrize

    # cmds_random[2], cmds_random[3] = check_lip_low(cmds_random[2], cmds_random[3])
    cmds_random[0],cmds_random[1],cmds_random[2],cmds_random[3] = check_lip_upper(cmds_random[0],cmds_random[1],cmds_random[2],cmds_random[3],cmds_random[5])
    cmds_random[4] = cmds_random[3]
    cmds_random[6] = cmds_random[5]
    cmds_random[9] = cmds_random[8]
    cmds_random[11] = cmds_random[10]


    if only_mouth:
        cmds_random[8:] = restart_face[8:]


    return cmds_random


def move_all(target_cmds, interval=50, with_eyes = True):
    if with_eyes:
        num_motors = len(all_motors)
    else:
        num_motors = len(all_motors)-4

    # Get current motor joint angles:
    curr = np.zeros(num_motors)
    for i in range(num_motors):
        curr[i] = all_motors[i].norm_v_cur

    traj = np.linspace(curr, target_cmds, num=interval+1, endpoint=True)
    # execute the commands:
    for i in range(1,interval+1):
        for j in range(num_motors):
            val = traj[i][j]
            all_motors[j].norm_act(val)
        time.sleep(0.004)
        # time.sleep(0.001)



def eyes_move_2_traget(l_point, r_point):
    step = 1

    l_loc = (0.5 - l_point[0]) * abs(0.5 - l_point[0]) * step + l_eye_yaw.norm_v_cur
    r_loc = (0.5 - r_point[0]) * abs(0.5 - r_point[0]) * step + r_eye_yaw.norm_v_cur
    l_loc = np.clip(l_loc, 0, 1)
    r_loc = np.clip(r_loc, 0, 1)

    l_loc_p = (0.5 - l_point[1]) * abs(0.5 - l_point[1]) * step + l_eye_pitch.norm_v_cur
    r_loc_p = (0.5 - r_point[1]) * abs(0.5 - r_point[1]) * step + r_eye_pitch.norm_v_cur
    l_loc_p = np.clip(l_loc_p, 0, 1)
    r_loc_p = np.clip(r_loc_p, 0, 1)


    l_eye_yaw.norm_act(l_loc)
    r_eye_yaw.norm_act(r_loc)
    l_eye_pitch.norm_act(l_loc_p)
    r_eye_pitch.norm_act(r_loc_p)

def control_face():
    cmd_lip_up = 1
    cmd_lip_up_warp = 1
    cmd_lip_down = 0.1
    cmd_r_corner_up = 0.5
    cmd_r_corner_low = 0.9
    jaw = 0.7

    target_cmds = [cmd_lip_up, cmd_lip_up_warp, cmd_lip_down, cmd_r_corner_up, cmd_r_corner_up,
                   cmd_r_corner_low, cmd_r_corner_low, jaw] + [0.4286, 0.4286, 0.6667, 0.6667]

    target_cmds = random_cmds(reference=target_cmds, noise=0.2, only_mouth=True)
    move_all(target_cmds)
    time.sleep(0.4)


def eyes_lid():
    for i in range(10000):
        l_upper_eyelid.norm_act(0.5 * np.cos((i + 1) / 400 * 2 * np.pi) + 0.5)
        r_upper_eyelid.norm_act(0.5 * np.cos((i + 1) / 400 * 2 * np.pi) + 0.5)
        l_lower_eyelid.norm_act(0.5 * np.cos((i + 1) / 400 * 2 * np.pi + np.pi) + 0.5)
        r_lower_eyelid.norm_act(0.5 * np.cos((i + 1) / 400 * 2 * np.pi + np.pi) + 0.5)
        time.sleep(0.01)


def blink():
    l_upper_eyelid.norm_act(0)
    r_upper_eyelid.norm_act(0)
    l_lower_eyelid.norm_act(1)
    r_lower_eyelid.norm_act(1)
    time.sleep(0.2)   
    l_upper_eyelid.norm_act(1)
    r_upper_eyelid.norm_act(1)
    l_lower_eyelid.norm_act(0)
    r_lower_eyelid.norm_act(0)  

# 10 steps per blink
def blink_seg(ti, c = 10, n_step= 8):
    if ti <= 8:
        l_upper_eyelid.norm_act(1-ti/n_step)
        r_upper_eyelid.norm_act(1-ti/n_step)
        l_lower_eyelid.norm_act(ti/n_step)
        r_lower_eyelid.norm_act(ti/n_step)
    # elif (ti >3) and (ti<6):
        # nothing
    elif ti>= c:
        l_upper_eyelid.norm_act(  (ti-c)/n_step)
        r_upper_eyelid.norm_act(  (ti-c)/n_step)
        l_lower_eyelid.norm_act(1-(ti-c)/n_step)
        r_lower_eyelid.norm_act(1-(ti-c)/n_step)  

def random_move(restf, scale_range = 0.1,loop_time =50,only_mouth=True):

    blink_count_threshold = 9
    blink_count = 0
    blink_flag = False

    for i in range(loop_time):
        print(i)
        target_cmds = random_cmds(reference=restf, noise=scale_range, only_mouth=only_mouth)


        print(target_cmds)
        move_all(target_cmds,interval=50)
        # time.sleep(0.5)

restart_face = [0.1, 0.0, 0.55556, 0.42857, 0.5, 0.32353, 0.24286, 1.0, 0.42857, 0.42857, 0.66667, 0.66667]
restart_face0 = [0.1, 0.0, 0.55556, 0.42857, 0.5, 0.32353, 0.24286, 0.5, 0.42857, 0.42857, 0.66667, 0.66667]  #0.3 open mouth
smile_face = [0.8, 0, 0.3,  0.8, 0.8, 0.6, 0.6, 0.8, 0.4286, 0.4286, 0.6667, 0.6667] # 0.5
smile_face0 = [0.8, 0, 0.3,  0.8, 0.8, 0.6, 0.6, 0.4, 0.4286, 0.4286, 0.6667, 0.6667] # 0.3 open mouth
pout_face = [0, 0.8, 1, 0.1, 0.1, 0.9, 0.9, 1.0, 0.42857, 0.42857, 0.66667, 0.66667] # 0.3
# pout_face0 = [0, 0.8, 1, 0.1, 0.1, 0.9, 0.9, .6, 0.42857, 0.42857, 0.66667, 0.66667] # 0.3

ref_face_list = [restart_face,restart_face0,smile_face,smile_face0,pout_face]

noise_list = [0.3,0.3,0.5,0.3,0.3,0.3]
 
wired_face = [0.01833095 ,0.82924096, 0.44764508, 0.      ,   0.   ,      0.91063554, 0.91063554 ,1.   ,      0.42857  ,  0.42857 ,   0.66667  ,  0.66667    ]

# combin_face = [resting_face,smile_face,upper_teeth]
from scipy.signal import savgol_filter

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20,10))
    plt.subplots_adjust(left=0.5)

    # Some initial data
    index_d_motor = [0,1,2,3,5,7]
    initial_data = np.asarray(restart_face)[index_d_motor]


    bars = plt.bar(range(1, 7), initial_data)
    plt.ylim(0,1)
    x_label = ['Upper Lip', 'Upper Lip Warp', 'Lower Lip', 'Upper Corner', 'Lower Corner', 'Jaw']

    # Vertical Slider
    amp_slider_list = []
    for i in range(6):
        axamp = plt.axes([ 0.05 + 0.07 * i, 0.1, 0.02, 0.8])
        amp_slider_list.append(Slider(axamp, x_label[i], 0., 1.0, valinit=initial_data[i], orientation="vertical"))


    # Update function
    def update(val):
        action_list = []
        for i in range(6):
            amplitude = amp_slider_list[i].val
            action_list.append(amplitude)
            bars[i].set_height(amplitude)

        fig.canvas.draw_idle()
        action_list = np.hstack((action_list[:4],action_list[3:4],action_list[4:5],action_list[4:5],action_list[5:6]))
        move_all(action_list,with_eyes=False,interval=10)

        

    # Register the update function with the slider
    for i in range(6):
        amp_slider_list[i].on_changed(update)

    # Create a list to store values
    stored_values = []

    # Button press handler
    def on_button_press(event):
        current_values = [slider.val for slider in amp_slider_list]
        stored_values.append(current_values)
        print(f"Stored values{len(current_values)}:", current_values)

    # Add a button
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    button = Button(ax_button, 'Store Values')
    button.on_clicked(on_button_press)


    # Show the plot
    plt.show()
    np.savetxt('log_action.csv',stored_values)

    quit()






    # # load_cmd = np.loadtxt(f'../data/true-sweep-2/9.csv') #vocal-sweep-7
    # # load_cmd = np.hstack((load_cmd[:,:4],load_cmd[:,3:4],load_cmd[:,4:5],load_cmd[:,4:5],load_cmd[:,5:6]))
    # load_cmd = np.loadtxt(f'../../data1213/action.csv')


    # for i in range(1000):
    #     a = input()
    #     a = int(a)
    #     # target_cmds = [ 0.621,  0.004,  0.305,  0.406, 0.406, -0.135, -0.135,  0.966]
    #     # target_cmds = [ 0.621,  0.004,  0.305,  0.406, 0.406, -0.135, -0.135,  0.966]
    #     target_cmds = load_cmd[a]
    #     print(target_cmds)
    #     for j in range(8):
    #         target_cmds[j] = np.clip(target_cmds[j],0,1)
    #         all_motors[j].norm_act(target_cmds[j])
    # quit()


    ##########################################
    ############# Abnormal test: #############
    ##########################################

    # for i in range(100):
    #     print('--------------------------------------------------')
    #     wired_face_modify = np.copy(np.asarray(wired_face))

    #     a = input()
    #     a = float(a)

    #     wired_face_modify[0] = a
    #     print("original: ")
    #     print(wired_face_modify)
    #     wired_face_modify[0],wired_face_modify[1],wired_face_modify[2],wired_face_modify[3] = check_lip_upper(wired_face_modify[0],wired_face_modify[1],wired_face_modify[2],wired_face_modify[3],wired_face_modify[5])
    #     wired_face_modify[4] = wired_face_modify[3]
    #     move_all(wired_face_modify)
    #     print(wired_face_modify)
    # # Save resting face position in normed space
    # quit()
    ##########################################
    ############ Random babbling #############
    ##########################################

    # np.random.seed(3)

    # restart_face = []
    # for m in all_motors:
    #      restart_face.append(round(m.norm_v_cur,5))
    # print(restart_face)
    
    # move_all(restart_face)
    
    # scale = 0.3
    # random_move(restart_face, scale,loop_time=1000,only_mouth=True)

    ##########################################
    ########## Run commands (Record)##########
    ##########################################

    # import os
    # time_interval = 1/30
    # idx_cmds = 0

    # filter_flag = False
    # model_name = 'denim-dawn-82'
    # load_cmd = np.loadtxt(f'../data/{model_name}/{idx_cmds}.csv')

    # # model_name = "nn_100" #'eager-sweep-2' #'fanciful-sweep-1' 
    # # load_cmd = np.loadtxt(f'../data/{model_name}/cmds_{idx_cmds}.csv')
    # load_cmd = np.hstack((load_cmd[:,:4],load_cmd[:,3:4],load_cmd[:,4:5],load_cmd[:,4:5],load_cmd[:,5:6]))

    # load_cmd = np.loadtxt('../data/nvidia/en_1_cmds.csv')
    # load_cmd = np.loadtxt('../data/nvidia/fs2_nn_cmds.csv')

    ####### After collecting data, shift the index ##########
    # load_cmd = np.loadtxt('../../data1124/action.csv')
    # copy_first = np.copy(load_cmd[:2])
    # load_cmd = np.vstack((copy_first,load_cmd[1:-1]))
    # np.savetxt('../data/action_tuned.csv',load_cmd)
    # for i in range(100):
    #     a_id = int(input())
    #     target_cmds = load_cmd[a_id]
    #     # target_cmds = ref_face_list[i%6]
    #     for j in range(8):
    #         all_motors[j].norm_act(target_cmds[j])
    # quit()
    ########################

    # # visualize commands
    # load_cmd_ori = np.copy(load_cmd)
    # for j in range(1):
    #     fig, axs = plt.subplots(9)
    #     fig.suptitle('motor command plots')
    #     for i in range(9):
    #         window = 5
    #         order = 2
    #         load_cmd_filt[:,i] = savgol_filter(load_cmd_filt[:,i], window, order) # window size 51, polynomial order 3
    #         axs[i].plot(list(range(len(load_cmd_ori[:,i]))),load_cmd_ori[:,i],label='raw')
    #         axs[i].plot(list(range(len(load_cmd_filt[:,i]))),load_cmd_filt[:,i],label='filtered')
    #     plt.legend()
    #     plt.savefig('../signal_processing_plots/savgol_%d_%d'%(window,order),dpi = 300)
    #     plt.clf()
    # quit()
    
    # Camera Record: 
    mode = 3

    # for i in range(8):
    #     load_cmd_filt[:,i] = savgol_filter(load_cmd_filt[:,i], window, order) # window size 51, polynomial order 3
    # time.sleep(1)

    if filter_flag:
        load_cmd_filt = np.copy(load_cmd)
            # Smooth:
        window = 5 #7 #13
        order = 2 #2 #3
        load_cmd_filt= savgol_filter(load_cmd_filt, window, order) # window size 51, polynomial order 3
    
        load_cmd = load_cmd_filt

    if mode == 0:
        time0 = time.time()
        eyelid_traj_id = 0
        blink_count_threshold = 105
        blink_count = 0
        blink_flag = False
        for i in range(len(load_cmd)):
            
            # A = int(input())
            target_cmds = load_cmd[i]
            print(target_cmds)
            # Mouth movements
            for j in range(8):
                target_cmds[j] = np.clip(target_cmds[j],0,1)
                all_motors[j].norm_act(target_cmds[j])

            ########## blink ###########
            # if blink_count > blink_count_threshold:
            #     blink_flag = True
            #     blink_count = 0

            # if blink_flag:
            #     blink_seg(blink_count)
            #     if blink_count == 18:
            #         blink_count = 0
            #         blink_flag = False
            # blink_count +=1

            time_used = time.time()-time0
            if time_used<time_interval:
                time.sleep(time_interval-time_used)
            else:
                print('NOT REALTIME')
            time0 = time.time()
            
    elif mode == 1:
        print('record mode')
        #   RECORD A VIDEO
        from collect_data import *
        from realtime_landmark import *

        cap = VideoCapture(4)

        # get cap property
        frame_width = cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        frame_height = cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        pcf = PCF(
            near=1,
            far=10000,
            frame_height=frame_height,
            frame_width=frame_width,
            fy=camera_matrix[1, 1]
        )

        # cv2.namedWindow("landmarks")
        # cv2.createTrackbar("vert", "landmarks", 180, 360, do_nothing)
        # cv2.createTrackbar("hori", "landmarks", 180, 360, do_nothing)
        img_i = 0
        r_lmks_logger = []
        m_lmks_logger = []
        action_logger = []

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for i in range(len(load_cmd)):
                if img_i == 0:
                    for _ in range(100):
                        image = cap.read()

                target_cmds = load_cmd[i]

                # execute the commands:
                for j in range(9):
                    target_cmds[j] = np.clip(target_cmds[j],0,1)
                    all_motors[j].norm_act(target_cmds[j])

                time.sleep(0.03)

                image = cap.read()
                # if not success:
                #     print("Ignoring empty camera frame.")
                #     # If loading a video, use 'break' instead of 'continue'.
                #     continue

                image_show, raw_lmks, m_lmks = render_img(image, face_mesh, pcf)
                # r_lmks_logger.append(raw_lmks)
                m_lmks_logger.append(m_lmks)

                # SAVE
                cv2.imwrite('../gpt_demo_output/img/%d.png' % img_i, image_show)
                img_i += 1
                # if img_i % 20 == 0:
                    # np.save('../dataset/resting_r_lmks.npy', np.asarray(r_lmks_logger))

                # cv2.imshow('landmarks', image_show)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            np.save('../gpt_demo_output/en1_m_lmks.npy', np.asarray(m_lmks_logger))
        cap.cap.release()

    elif mode == 2:
        import sys
        sys.path.append('../../test/')
        from camera_test import *
        if filter_flag:
            video_log_path = f'../data/{model_name}/filter/output{idx_cmds}.mp4'
        else:
            video_log_path = f'../data/{model_name}/output{idx_cmds}.mp4'
        recorder = VideoRecorder(cam_id = 0,output_file=video_log_path)
        recorder.start_recording()
        time.sleep(4)
        time0 = time.time()
        for i in range(len(load_cmd)):
            
            target_cmds = load_cmd[i]
            print(target_cmds)
            # Mouth movements
            for j in range(8):
                target_cmds[j] = np.clip(target_cmds[j],0,1)
                all_motors[j].norm_act(target_cmds[j])

            time_used = time.time()-time0
            if time_used<time_interval:
                time.sleep(time_interval-time_used)
            else:
                print('NOT REALTIME')
            time0 = time.time()

        recorder.stop_recording()

    elif mode == 3:
        
        cap = cv2.VideoCapture(0)
        os.makedirs(f'../data/{model_name}/img{idx_cmds}/',exist_ok=True)
        for img_i in range(len(load_cmd)):
            if img_i == 0:
                for _ in range(100):
                    ret,image = cap.read()

            target_cmds = load_cmd[img_i]

            # execute the commands:
            for j in range(8):
                target_cmds[j] = np.clip(target_cmds[j],0,1)
                all_motors[j].norm_act(target_cmds[j])

            time1 =time.time()
            for _ in range(30):
                ret,image = cap.read()
            time2 =time.time()
            print(time2-time1)
            # SAVE
            cv2.imwrite(f'../data/{model_name}/img%d/%d.png' % (idx_cmds, img_i), image)



    # only move jaw
    else:
        import sys
        sys.path.append('../../test/')
        from camera_test import *
        if filter_flag:
            video_log_path = f'../data/{model_name}/filter/output{idx_cmds}.mp4'
        else:
            video_log_path = f'../data/{model_name}/output{idx_cmds}.mp4'
        recorder = VideoRecorder(cam_id = 0,output_file=video_log_path)
        recorder.start_recording()
        time.sleep(4)
        time0 = time.time()
        for i in range(len(load_cmd)):
            target_cmds = np.clip(load_cmd[i],0,1)
            all_motors[7].norm_act(target_cmds)

            time_used = time.time()-time0
            if time_used<time_interval:
                time.sleep(time_interval-time_used)
            else:
                print('NOT REALTIME')
            time0 = time.time()

        recorder.stop_recording()

