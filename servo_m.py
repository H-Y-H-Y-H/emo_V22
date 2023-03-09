from adafruit_servokit import ServoKit
import numpy as np
import random
import time


########################################
################# Face #################
########################################
pi = 3.14159
kit = [ServoKit(channels = 16, address = 0x41), 
       ServoKit(channels = 16, address = 0x42)]

class Actuator(object):
    def __init__(self, idx_tuple, min_angle, max_angle, init_angle, inverse_flag = False):
        self.idx = idx_tuple      # data format: (kit number, channel)
        self.v_min = min_angle
        self.v_max = max_angle
        self.ranges = self.v_max - self.v_min
        self.v_cur = init_angle
        self.norm_v_cur = (self.v_cur - self.v_min)/self.ranges
        if inverse_flag == True:
            self.norm_v_cur = 1 - self.norm_v_cur
        self.norm_v_init = self.norm_v_cur
        self.inverse_value = inverse_flag
        self.norm_act(self.norm_v_init)

    # Input normalized cmds: [0,1]
    def norm_act(self, norm_cmd):
        if norm_cmd > 1.0 or norm_cmd < 0:
            print(norm_cmd,self.idx,"Norm cmd is not right, it should between 0 ~ 1.")
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
            print(cmd,"Cmd is out of action space")
            quit()
        else:
            kit[self.idx[0]].servo[self.idx[1]].angle = cmd
        self.v_cur = cmd
        self.norm_v_cur = (cmd - self.v_min)/self.ranges


    def ret(self):
        kit[self.idx[0]].servo[self.idx[1]].angle = self.v_min
        self.v_cur = self.v_min 
        self.norm_v_cur = 0

def check_lip_low(cmd_lip_down, cmd_lip_down_warp):

    # regenerate the lip values.
    if cmd_lip_down+cmd_lip_down_warp<0.8:
        cmd_lip_down_warp = np.random.uniform((0.8-cmd_lip_down),1)

    return cmd_lip_down, cmd_lip_down_warp


# add value 

lip_up      = Actuator(idx_tuple = (0,0), min_angle = 70, max_angle = 95, init_angle = 95, inverse_flag=True)
lip_up_warp = Actuator(idx_tuple = (0,1), min_angle = 70, max_angle = 100,  init_angle = 70 )

lip_down = Actuator(idx_tuple = (0,4), min_angle = 65, max_angle = 88 , init_angle = 88 )
lip_down_warp = Actuator(idx_tuple = (0,5), min_angle = 100, max_angle = 130 , init_angle = 130)
#  lip_down_warp_cmds = np.random.uniform((0.8-lip_down_cmds),1)

r_corner_up = Actuator(idx_tuple = (0,2), min_angle = 40, max_angle = 110 , init_angle = 65 )
l_corner_up = Actuator(idx_tuple = (0,7), min_angle = 20, max_angle = 90 , init_angle = 65,inverse_flag = 1 )

r_corner_low = Actuator(idx_tuple = (0,3), min_angle = 20, max_angle = 100 , init_angle = 62  )
l_corner_low = Actuator(idx_tuple = (0,6), min_angle = 90, max_angle = 170 , init_angle = 127,inverse_flag =1  )

r_eye_yaw = Actuator(idx_tuple = (0,8), min_angle = 65, max_angle = 125 , init_angle = 95  )
l_eye_yaw = Actuator(idx_tuple = (1,7), min_angle = 60, max_angle = 120 , init_angle = 90  )

r_eye_pitch = Actuator(idx_tuple = (0,9), min_angle = 68, max_angle = 98 , init_angle = 83  )
l_eye_pitch = Actuator(idx_tuple = (1,6), min_angle = 78, max_angle = 108 , init_angle = 93, inverse_flag = True  )

r_upper_eyelid = Actuator(idx_tuple = (0,11),min_angle = 80, max_angle = 163 , init_angle = 80, inverse_flag= True  )
l_upper_eyelid = Actuator(idx_tuple = (1,4), min_angle = 18, max_angle = 100 , init_angle = 100  )

r_lower_eyelid = Actuator(idx_tuple = (0,10), min_angle = 93, max_angle = 143 , init_angle = 93   )
l_lower_eyelid = Actuator(idx_tuple = (1,5), min_angle = 40, max_angle = 95 , init_angle = 95, inverse_flag = True     )

jaw = Actuator(idx_tuple = (1,15), min_angle = 40, max_angle = 90, init_angle = 90)

neck_mode = False

if neck_mode == True:
    neck_yaw = Actuator(idx_tuple = (1,14), min_angle = 20, max_angle = 160, init_angle = 90)
    neck_roll = Actuator(idx_tuple = (1,13), min_angle = 90, max_angle = 120, init_angle = 105)
    neck_pitch= Actuator(idx_tuple = (1,12), min_angle = 20, max_angle = 60, init_angle = 40, inverse_flag = 1)
    all_motors = [lip_up, lip_up_warp, lip_down, lip_down_warp, r_corner_up, l_corner_up, r_corner_low, l_corner_low, jaw,neck_roll,neck_pitch,neck_yaw ]
    all_motors = [lip_up, lip_up_warp, lip_down, lip_down_warp, r_corner_up, l_corner_up, r_corner_low, l_corner_low, jaw,neck_roll,neck_pitch,neck_yaw ]
else:
    all_motors = [lip_up, lip_up_warp, lip_down, lip_down_warp, r_corner_up, l_corner_up, r_corner_low, l_corner_low, jaw ]
    all_motors = [lip_up, lip_up_warp, lip_down, lip_down_warp, r_corner_up, l_corner_up, r_corner_low, l_corner_low, jaw ]



def random_move(interval = 50):
    num_motors = len(all_motors)
    curr = np.zeros(num_motors)

    # Get current motor joint angles:
    for i in range(num_motors):
        curr[i] = all_motors[i].norm_v_cur
    
    # Generate random movement and check lower lips
    cmds_random = np.random.sample(num_motors)
    cmds_random[5] = cmds_random[4]
    cmds_random[7] = cmds_random[6]
    cmds_random[2], cmds_random[3] = check_lip_low(cmds_random[2], cmds_random[3])

    print("execute random commands: ", cmds_random)
    # Generate trajectories
    traj = np.linspace(curr, cmds_random, num=interval, endpoint=True)

    # execute the commands:
    for i in range(interval):
        for j in range(num_motors):
            val = traj[i][j]

            all_motors[j].norm_act(val)
            time.sleep(0.001)


def random_move_neck():
    neck_control = [neck_roll,neck_pitch,neck_yaw]
    n_steps = 100
    move_times = 10

    for i in range(move_times):
        target_pos = np.random.sample(3)
        if i == move_times -1:
            target_pos = np.ones(3)*0.5
        print(target_pos)
        for j in range(n_steps):
            cur_pos = np.asarray([neck_roll.norm_v_cur,neck_pitch.norm_v_cur,neck_yaw.norm_v_cur])

            neck_values = (target_pos - cur_pos)*0.05 + cur_pos
            print(neck_values)
        
            for j in range(len(neck_control)):
                neck_control[j].norm_act(neck_values[j])
                time.sleep(0.01)
    
def random_smile():
        # smile_normal_random = np.random.normal(loc = smile_2, scale = scale_range)
        # smile_normal_random = np.clip(smile_normal_random,0,1)

        smile_normal_random = np.random.sample(9)

        # Symentric:
        smile_normal_random[4],smile_normal_random[6] = smile_normal_random[5],smile_normal_random[7]
        # Check upper lips
        smile_normal_random[0],smile_normal_random[1] = check_lip_top(smile_normal_random[0],smile_normal_random[1])

        # Add neck
        # neck_pos = np.random.normal(loc = (0.5,0.7,0.5),scale = (0.2,0.2,0.1))
        # neck_pos = np.clip(neck_pos, 0, 1)
        # smile_normal_random = np.hstack((smile_normal_random,neck_pos))

        move_all(smile_normal_random)
        time.sleep(0.5)


def eyes_module(i):
    print(i)
    # r_eye_yaw.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)
    # l_eye_yaw.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)
    # l_upper_eyelid.norm_act(0.5*np.cos((i+1)/400 * 2*np.pi)+0.5)
    # r_upper_eyelid.norm_act(0.5*np.cos((i+1)/400 * 2*np.pi)+0.5)
    # l_lower_eyelid.norm_act(0.5*np.cos((i+1)/400 * 2*np.pi+np.pi)+0.5)
    # r_lower_eyelid.norm_act(0.5*np.cos((i+1)/400 * 2*np.pi+np.pi)+0.5)
    # l_eye_pitch.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)
    # r_eye_pitch.norm_act(0.5*np.sin((i+1)/400 * 2*np.pi)+0.5)
    time.sleep(0.01)


if __name__ == "__main__":

# Save resting face position in normed space
    np.random.seed(1)

    resting_face = []
    for m in all_motors:
        resting_face.append(m.norm_v_cur)
        print(m.norm_v_cur)
    resting_face = [0.0,0.0,1.0,1.0,0.357,0.357,0.525,0.5375,1.0]
    
    scale_range = 1

    for i in range(100):
        random_move()
        time.sleep(0.2)
        # eyes_module(i)
        # print(i)
        # random_smile()


        # print(v)
        # lip_down_warp.act(v)
        
        # test_v = 0.5*np.sin(i* np.pi/2 /100)+0.5
        # test.norm_act(test_v)
        # time.sleep(0.005)


    # for i in range(100):
    #     time.sleep(0.3)

    #     cmd_lip_up_warp, cmd_lip_up = np.random.sample(size= 2)
    #     cmd_lip_up_warp, cmd_lip_up = check_lip_top(cmd_lip_up_warp, cmd_lip_up)
    #     lip_up_warp.norm_act(cmd_lip_up_warp)
    #     lip_up.norm_act(cmd_lip_up)
        


