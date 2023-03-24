from adafruit_servokit import ServoKit
import numpy as np
import random
import time

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


def check_lip_low(cmd_lip_down, cmd_lip_down_warp):
    # regenerate the lip values.
    if cmd_lip_down + cmd_lip_down_warp < 0.8:
        cmd_lip_down_warp = np.random.uniform((0.8 - cmd_lip_down), 1)
        print('reproduce cmd_lip_down_warp ', )

    return cmd_lip_down, cmd_lip_down_warp


# add value

lip_up = Actuator(idx_tuple=(0, 0), min_angle=70, max_angle=103, init_angle=103, inverse_flag=True)
lip_up_warp = Actuator(idx_tuple=(0, 1), min_angle=70, max_angle=100, init_angle=70)

lip_down = Actuator(idx_tuple=(0, 4), min_angle=65, max_angle=96, init_angle=90)
lip_down_warp = Actuator(idx_tuple=(0, 5), min_angle=100, max_angle=130, init_angle=105, inverse_flag=True)
#  lip_down_warp_cmds = np.random.uniform((0.8-lip_down_cmds),1)

r_corner_up = Actuator(idx_tuple=(0, 2), min_angle=40, max_angle=110, init_angle=70)
l_corner_up = Actuator(idx_tuple=(0, 7), min_angle=20, max_angle=90, init_angle=60, inverse_flag=1)

r_corner_low = Actuator(idx_tuple=(0, 3), min_angle=20, max_angle=100, init_angle=62)
l_corner_low = Actuator(idx_tuple=(0, 6), min_angle=90, max_angle=170, init_angle=128, inverse_flag=1)

r_eye_yaw = Actuator(idx_tuple=(0, 8), min_angle=65, max_angle=125, init_angle=95)
l_eye_yaw = Actuator(idx_tuple=(1, 7), min_angle=60, max_angle=120, init_angle=90)

r_eye_pitch = Actuator(idx_tuple=(0, 9), min_angle=68, max_angle=98, init_angle=83)
l_eye_pitch = Actuator(idx_tuple=(1, 6), min_angle=78, max_angle=108, init_angle=93, inverse_flag=True)

r_upper_eyelid = Actuator(idx_tuple=(0, 11), min_angle=80, max_angle=163, init_angle=80, inverse_flag=True)
l_upper_eyelid = Actuator(idx_tuple=(1, 4), min_angle=18, max_angle=100, init_angle=100)

r_lower_eyelid = Actuator(idx_tuple=(0, 10), min_angle=93, max_angle=143, init_angle=93)
l_lower_eyelid = Actuator(idx_tuple=(1, 5), min_angle=40, max_angle=95, init_angle=95, inverse_flag=True)

r_inner_eyebrow = Actuator(idx_tuple=(0, 14), min_angle=65, max_angle=100, init_angle=85, inverse_flag=True)
l_inner_eyebrow = Actuator(idx_tuple=(1, 1), min_angle=75, max_angle=110, init_angle=90)

r_outer_eyebrow = Actuator(idx_tuple=(0, 15), min_angle=80, max_angle=125, init_angle=110)
l_outer_eyebrow = Actuator(idx_tuple=(1, 0), min_angle=60, max_angle=105, init_angle=75, inverse_flag=True)

jaw = Actuator(idx_tuple=(1, 15), min_angle=40, max_angle=90, init_angle=90)

neck_mode = False
neck_yaw = Actuator(idx_tuple=(1, 14), min_angle=20, max_angle=160, init_angle=85)

if neck_mode == True:
    neck_roll = Actuator(idx_tuple=(1, 13), min_angle=90, max_angle=120, init_angle=105)
    neck_pitch = Actuator(idx_tuple=(1, 12), min_angle=20, max_angle=60, init_angle=40, inverse_flag=1)
    all_motors = [lip_up, lip_up_warp, lip_down, lip_down_warp, r_corner_up, l_corner_up, r_corner_low, l_corner_low,
                  jaw,
                  r_inner_eyebrow, l_inner_eyebrow, r_outer_eyebrow, l_outer_eyebrow,
                  neck_roll, neck_pitch, neck_yaw]
else:
    all_motors = [lip_up, lip_up_warp, lip_down, lip_down_warp,
                  r_corner_up, l_corner_up, r_corner_low, l_corner_low,
                  jaw,
                  r_inner_eyebrow, l_inner_eyebrow, r_outer_eyebrow, l_outer_eyebrow]


def random_cmds(reference=None, noise=0.2, only_mouth=True):
    num_motors = len(all_motors)

    if reference != None:
        cmds_random = np.random.normal(reference, scale=noise)
    else:
        # Generate random movement and check lower lips
        cmds_random = np.random.sample(num_motors)

    # Symmetrize
    cmds_random[5] = cmds_random[4]
    cmds_random[7] = cmds_random[6]
    cmds_random[10] = cmds_random[9]
    cmds_random[12] = cmds_random[11]
    cmds_random[2], cmds_random[3] = check_lip_low(cmds_random[2], cmds_random[3])

    if only_mouth:
        cmds_random[9:] = resting_face[9:]

    cmds_random = np.clip(cmds_random, 0, 1)

    return cmds_random


def move_all(target_cmds, interval=50):
    num_motors = len(all_motors)
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
        time.sleep(0.008)


def eyes_move_2_traget(l_point, r_point):
    step = 1

    l_loc = (0.5 - l_point) * abs(0.5 - l_point) * step + l_eye_yaw.norm_v_cur
    r_loc = (0.5 - r_point) * abs(0.5 - r_point) * step + r_eye_yaw.norm_v_cur
    l_loc = np.clip(l_loc, 0, 1)
    r_loc = np.clip(r_loc, 0, 1)
    l_eye_yaw.norm_act(l_loc)
    r_eye_yaw.norm_act(r_loc)
    print(l_loc, r_loc)
    # r_eye_yaw.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)
    # l_eye_yaw.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)

    # l_eye_pitch.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)
    # r_eye_pitch.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)
    # time.sleep(0.01)


def control_face():
    cmd_lip_up = 1
    cmd_lip_up_warp = 1
    cmd_lip_down = 0.1
    cmd_lip_down_warp = 0
    cmd_r_corner_up = 0.5
    cmd_r_corner_low = 0.9
    jaw = 0.7

    target_cmds = [cmd_lip_up, cmd_lip_up_warp, cmd_lip_down, cmd_lip_down_warp, cmd_r_corner_up, cmd_r_corner_up,
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



resting_face = [0.0,
0.0,
0.8064516129032258,
0.8333333333333334,
0.42857142857142855,
0.4285714285714286,
0.525,
0.525,
1.0,
0.4285714285714286,
0.42857142857142855,
0.6666666666666666,
0.6666666666666667]

smile_face = [0.5, 0, 1, 1, 1, 1, 0.6, 0.6, 0.8, 0.4286, 0.4286, 0.6667, 0.6667]
upper_teeth = [1, 0, 0.1, 0, 0.5, 0.5, 0.9, 0.9, 0.7, 0.4286, 0.4286, 0.6667, 0.6667]
lower_teeth = [0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.8, 0.4286, 0.4286, 0.6667, 0.6667]
pout_face = [0.5, 1, 1, 0.0, 0.1, 0.1, 1, 1, 0.8, 0.4286, 0.4286, 0.6667, 0.6667]

combin_face = [resting_face,smile_face,upper_teeth]
if __name__ == "__main__":

    # Save resting face position in normed space
    np.random.seed(1)

    resting_face = []
    for m in all_motors:
        resting_face.append(m.norm_v_cur)
        print(m.norm_v_cur)

    scale_range = 1

    # for i in range(10,20):
    #     target_cmds = random_cmds(reference=resting_face, noise=0.5, only_mouth=True)
    #     move_all(action_list[i])
    #     time.sleep(5)

        # print(v)
        # lip_down_warp.act(v)

        # test_v = 0.5*np.sin(i* np.pi/2 /100)+0.5
        # test.norm_act(test_v)
        # time.sleep(0.005)
    load_cmd = np.load('../en-1_nn.npy')
    # load_cmd = np.load('../en-1_pred.npy')

    time0 = time.time()
    for i in range(len(load_cmd)):
        target_cmds = load_cmd[i]
        for j in range(len(target_cmds)):
            target_cmds[j] = np.clip(target_cmds[j],0,1)
            all_motors[j].norm_act(target_cmds[j])


        time_used = time.time()-time0
        time.sleep(0.04-time_used)
        time0 = time.time()



