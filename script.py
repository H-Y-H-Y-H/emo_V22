import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# Combine the dataset arr
# dataset_saved = '/Users/yuhang/Downloads/EMO_GPTDEMO/data0903/'
# dataset_pth3 = '/Users/yuhang/Downloads/EMO_GPTDEMO/data0903_2023seed/'
# dataset_pth4 = '/Users/yuhang/Downloads/EMO_GPTDEMO/data0903_2024seed/'
#
# data0_3 = np.load(dataset_pth3+'m_lmks.npy')
# data1_3 = np.load(dataset_pth3+'r_lmks.npy')
# data3_3 = np.loadtxt(dataset_pth3+'action.csv')
#
# data0_4 = np.load(dataset_pth4   +'m_lmks.npy')
# data1_4 = np.load(dataset_pth4   +'r_lmks.npy')
# data3_4 = np.loadtxt(dataset_pth4+'action.csv')
#
# data0 = np.concatenate((data0_3,data0_4))
# data1 = np.concatenate((data1_3,data1_4))
# data3 = np.concatenate((data3_3,data3_4))
# print(data0.shape)
# print(data3.shape)
# np.save(dataset_saved+'m_lmks.npy',   data0 )
# np.save(dataset_saved+'r_lmks.npy',   data1)
# np.savetxt(dataset_saved+'action.csv',data3)

# Combine the dataset img
# import shutil
# for i in range(6000):
#     datapath = '/Users/yuhang/Downloads/EMO_GPTDEMO/data0903_2024seed/img/%d.png'%i
#     savepath = '/Users/yuhang/Downloads/EMO_GPTDEMO/data0903/img/%d.png'%(i+6000)
#     shutil.copy(datapath,savepath)

# video_path = 'data/desktop/synced_video.avi'
# video_path = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/output0917_mimic(smooth_lmks)_3_13.mp4'
# save_path = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/gpt_demo_output0917/real_frame/'
# import cv2
# vidcap = cv2.VideoCapture(video_path)
# # vidcap.set(cv2.CAP_PROP_FPS, 25)
# success,image = vidcap.read()
# cv2.imwrite(save_path+"frame%d.png" % 0, image)
# count = 1
# success = True
# while success:
#   success,image = vidcap.read()
#   cv2.imwrite(save_path+"frame%d.png" % count, image)     # save frame as JPEG file
#   if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#       break
#   count += 1
#   print(count)


import cv2
import glob


def frames_2_video():
  # img_array = []
  # img_list = glob.glob('/Users/yuhan/PycharmProjects/EMO_GPTDEMO/data1105/img/*.png')
  # img_pth = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/data1201/img/'
  # frame_n = 10000
  # fps = 30
  # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  # out = cv2.VideoWriter('/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/data1201/data_%ds.mp4'%(frame_n//fps), fourcc, fps, (480, 480))
  method_name = 'wandering-sweep-1'
  idx = 9
  img_pth = f'/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/output_cmds/{method_name}/img{idx}/'
  frame_n = len(os.listdir(img_pth))
  fps = 30
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  out = cv2.VideoWriter(f'/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/output_cmds/{method_name}/{idx}.mp4', fourcc, fps, (480, 480))

  for i in range(frame_n):
    filename = img_pth+"/%d.png"%(i)

    img = cv2.imread(filename)[:,80:560]
    # img = img[:,80:560]

    print(filename)

    out.write(img)

  out.release()

# frames_2_video()
# quit()

def frames_video():
  img_array = []
  rank_num = 200
  d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/'

  img_pth = d_root+'desktop/NN(BL)_dataset/'
  img_list = os.listdir(img_pth+"/img(%d_m_lmks_mouth)/"%rank_num)
  print(len(img_list))
  for i in range(len(img_list)):
    filename = img_pth+"/img(%d_m_lmks_mouth)/%d.jpeg"%(rank_num,i)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    print(filename)

  out = cv2.VideoWriter(img_pth+'nn(dataset)_(%d_m_lmks_mouth).avi'%rank_num, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()


def visualize_dataset():
  import time
  dataset_pth = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/data0914/'
  filter_out = 3043
  dataset = np.load(dataset_pth + 'm_lmks.npy')[filter_out:]
  cv2.namedWindow("Robot")

  curr_id = 0
  for i in range(len(dataset)):
    t1 = time.time()
    if curr_id >= i:
      continue
    curr_id = i + np.random.randint(0,3)
    img = cv2.imread(dataset_pth+ "img/%d.png"%curr_id)

    cv2.imshow('Dataset_Review', img)
    if cv2.waitKey(1) == ord('q'):
      break
    t2 = time.time()
    if t2-t1 < (1/25):
      time.sleep(1/25-(t2-t1))

# visualize_dataset()


def video_2_frames(video_source,crop_flag=False):
  cap = cv.VideoCapture(video_source)
  list_img = []
  while cap.isOpened():
      ret, frame = cap.read()
      # if frame is read correctly ret is True
      if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break
      if crop_flag:
        frame = frame[:,80:560]
      cv.imshow('frame', frame)
      list_img.append(frame)
      if cv.waitKey(1) == ord('q'):
          break
  cap.release()
  cv.destroyAllWindows()
  return list_img


def cut_video():
  size = (480,480)
  method_name = 'wav_bl'#'nn_100' #'wav_bl'#'om'
  shift_frame = 3
  for demo_id in range(10):
    lmks_path = f'../EMO_GPTDEMO/robot_data/synthesized/lmks/m_lmks_{demo_id}.npy'
    length_f = len(np.load(lmks_path))
    print('length_f:',length_f)
    # video_source = '../EMO_GPTDEMO/robot_data/real_video/output%d.mp4'%demo_id
    video_source = f'../EMO_GPTDEMO/robot_data/output_cmds/{method_name}_video/output%d.mp4'%demo_id

    img_array = video_2_frames(video_source,crop_flag=True)
    img_array = np.asarray(img_array)[-length_f+shift_frame:]
    img_array = np.concatenate((img_array,img_array[-1:],img_array[-1:],img_array[-1:]))
    out = cv2.VideoWriter(f'../EMO_GPTDEMO/robot_data/output_cmds/{method_name}_video/{demo_id}.mp4',
                          cv2.VideoWriter_fourcc(*'MP4V'),
                          30,
                          size)

    for i in range(len(img_array)):
      out.write(img_array[i])
    out.release()

# cut_video()


def compare_lmks_dist():
  method_name = 'wav_bl'
  for demo_id in range(10):
    results = np.load(f'../EMO_GPTDEMO/output_cmds/{method_name}_video/m_lmks_{demo_id}.npy')
    label_lmks_path = f'../EMO_GPTDEMO/synthesized_target/om/lmks/m_lmks_{demo_id}.npy'
    label_lmks = np.load(label_lmks_path)

    error_list = [np.mean(np.abs(label_lmks - results))]
    for shif_length in range(1,10):
      error = np.mean(np.abs(label_lmks[:-shif_length] - results[shif_length:]))
      error_list.append(error)
    print(np.argmin(error_list),error_list)
# compare_lmks_dist()



from moviepy.editor import VideoFileClip, AudioFileClip

def combine_audio_video(audio_file_path, video_file_path, output_file_path):
    # Load the audio file
    audio_clip = AudioFileClip(audio_file_path)

    # Load the video file
    video_clip = VideoFileClip(video_file_path)

    # Set the audio of the video clip as the audio file
    final_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file
    final_clip.write_videofile(output_file_path, codec="libx264", audio_codec="aac")


# for idx in range(5,10):
#   method_name = 'om_video'
#   combine_audio_video(audio_file_path='../EMO_GPTDEMO/audio/emo/emo%d.wav'%idx,
#                       video_file_path=f'../EMO_GPTDEMO/robot_data/output_cmds/{method_name}/{idx}.mp4',
#                       output_file_path = f'../EMO_GPTDEMO/robot_data/output_cmds/{method_name}/%d(audio).mp4'%idx,
#                     )

# for idx in range(10):
#   method_name = 'nn_100'
#   combine_audio_video(audio_file_path='../EMO_GPTDEMO/audio/emo/emo%d.wav'%idx,
#                       video_file_path=f'../EMO_GPTDEMO/robot_data/output_cmds/{method_name}/{idx}.avi',
#                       output_file_path = f'../EMO_GPTDEMO/robot_data/output_cmds/{method_name}_%d.avi'%idx,
#                     )


lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

select_lmks_id = lips_idx + inner_lips_idx

def sidebyside(demo_id = 0):
  import matplotlib
  matplotlib.use('Agg')

  source_path = f'../EMO_GPTDEMO/robot_data/output_cmds/'
  method_names = ['syn', 'wandering-sweep-1',  'nn_100', 'wav_bl']
  save_path = f'../EMO_GPTDEMO/compare/'
  os.makedirs(save_path+f'demo{demo_id}/',exist_ok=True)
  video_frames_list = []
  for i in range(len(method_names)):
    met_n = method_names[i]
    video_path = source_path + met_n + f'_video/{demo_id}.mp4'
    frames = video_2_frames(video_path)
    video_frames_list.append(frames)

  img_list = []

  scale_change = (45/0.15)
  gt_lmks     = np.load(f'../EMO_GPTDEMO/robot_data/output_cmds/{method_names[0]}_video/lmks/m_lmks_{demo_id}.npy')*scale_change
  eager_lmks  = np.load(f'../EMO_GPTDEMO/robot_data/output_cmds/{method_names[1]}_video/lmks/m_lmks_{demo_id}.npy')*scale_change
  nn_400_lmks = np.load(f'../EMO_GPTDEMO/robot_data/output_cmds/{method_names[2]}_video/lmks/m_lmks_{demo_id}.npy')*scale_change
  wav_bl_lmks = np.load(f'../EMO_GPTDEMO/robot_data/output_cmds/{method_names[3]}_video/lmks/m_lmks_{demo_id}.npy')*scale_change

  gt_lmks     = gt_lmks     [:,select_lmks_id]
  eager_lmks  = eager_lmks  [:,select_lmks_id]
  nn_400_lmks = nn_400_lmks [:,select_lmks_id]
  wav_bl_lmks = wav_bl_lmks [:,select_lmks_id]

  dist_eager   = np.mean(np.abs(gt_lmks - eager_lmks) ,axis=(1,2))
  dist_nn_400  = np.mean(np.abs(gt_lmks - nn_400_lmks),axis=(1,2))
  dist_wav_bl = np.mean(np.abs(gt_lmks - wav_bl_lmks),axis=(1,2))

  # dist_eager  [:2] = dist_eager  [2]
  # dist_nn_400 [:2] = dist_nn_400 [2]
  # dist_wav_bl [:2] = dist_wav_bl [2]

  for f in range(len(video_frames_list[0])):
    plt.figure(figsize=(480 / 96, 480 / 96), dpi=96)
    plt.ylim(0,5)

    plt.plot(dist_eager,c = 'black',label='Overall L1 Distance')
    mean_dist = np.mean(dist_eager)
    plt.plot([0,len(dist_eager)], [mean_dist]*2,c='deepskyblue', linestyle='--', label =f'Mean L1 Distance')
    plt.scatter([f],[dist_eager[f]],c = 'r',label=f'Current Distance: {dist_eager[f]:.3f}')
    plt.xlabel("Frame Index")
    plt.ylabel("L1 Distance")
    plt.title(f"Mean L1 Landmarks Distance: {mean_dist:.3f}")
    plt.legend()
    plt.gca().figure.canvas.draw()
    # Convert to a NumPy array
    data = np.frombuffer(plt.gca().figure.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    eager_curve = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)


    plt.figure(figsize=(480 / 96, 480 / 96), dpi=96)
    plt.ylim(0,5)
    plt.plot(dist_nn_400,c = 'black',label='Overall L1 Distance')
    mean_dist = np.mean(dist_nn_400)
    plt.plot([0,len(dist_nn_400)], [mean_dist]*2,c='deepskyblue', linestyle='--', label =f'Mean L1 Distance')
    plt.scatter([f],[dist_nn_400[f]],c = 'r',label=f'Current Distance: {dist_nn_400[f]:.3f}')
    plt.xlabel("Frame Index")
    plt.ylabel("L1 Distance")
    plt.title(f"Mean L1 Landmarks Distance: {mean_dist:.3f}")
    plt.legend()
    plt.gca().figure.canvas.draw()
    # Convert to a NumPy array
    data = np.frombuffer(plt.gca().figure.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    nn_curve = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)


    plt.figure(figsize=(480 / 96, 480 / 96), dpi=96)
    plt.ylim(0,5)

    plt.plot(dist_wav_bl,c = 'black',label='Overall L1 Distance')
    mean_dist = np.mean(dist_wav_bl)
    plt.plot([0,len(dist_wav_bl)], [mean_dist]*2,c='deepskyblue', linestyle='--', label =f'Mean L1 Distance')
    plt.scatter([f],[dist_wav_bl[f]],c = 'r',label=f'Current Distance: {dist_wav_bl[f]:.3f}')
    plt.xlabel("Frame Index")
    plt.ylabel("L1 Distance")
    plt.title(f"Mean L1 Landmarks Distance: {mean_dist:.3f}")
    plt.legend()
    plt.gca().figure.canvas.draw()
    # Convert to a NumPy array
    data = np.frombuffer(plt.gca().figure.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    wav_bl_curve = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)
    ###############################

    img0 = video_frames_list[0][f]
    img1 = video_frames_list[1][f]
    img2 = video_frames_list[2][f]
    img3 = video_frames_list[3][f]

    img_1 = np.zeros_like(img0).astype(np.uint8)
    paddding = np.zeros((480,30,3))

    paddding = paddding.astype(np.uint8)

    img_com0 = np.hstack((img0,paddding,img1,paddding,img2,paddding,img3))
    img_com1 = np.hstack((img_1,paddding,eager_curve,paddding,nn_curve,paddding,wav_bl_curve))
    paddding_h = np.ones((30,img_com0.shape[1],3)).astype(np.uint8)

    img_com = np.vstack((img_com0,paddding_h,img_com1))

    # cv2.imshow('frame',data)
    cv2.imwrite(save_path+f'/demo{demo_id}/{f}.jpeg',img_com)
    img_list.append(img_com)
    imgshape = img_com.shape
    size = (imgshape[1],imgshape[0])

  out = cv2.VideoWriter(save_path+f'{demo_id}.mp4',
                        cv2.VideoWriter_fourcc(*'MP4V'),
                        30,
                        size)


  for i in range(len(img_list)):
    out.write(img_list[i])
  out.release()

  audio_path = f'../EMO_GPTDEMO/audio/emo/emo{demo_id}.wav'

  syn_video_path = save_path + f'{demo_id}.mp4'
  out_video_path = save_path + f'final_{demo_id}.mp4'

  # Usage
  combine_audio_video(audio_path, syn_video_path, out_video_path)

for i in range(9,10):
  sidebyside(i)

def debug_compare():
  method_name = 'nn_100'
  for demo_id in range(10):
    results = np.load(f'../EMO_GPTDEMO/robot_data/output_cmds/{method_name}_video/lmks/m_lmks_{demo_id}.npy')
    label_lmks_path = f'../EMO_GPTDEMO/robot_data/synthesized/lmks/m_lmks_{demo_id}.npy'
    label_lmks = np.load(label_lmks_path)

    error_list = [np.mean(np.abs(label_lmks - results))]
    for shif_length in range(1,10):
      error = np.mean(np.abs(label_lmks[:-shif_length] - results[shif_length:]))
      error_list.append(error)
    print(np.argmin(error_list),error_list)

# debug_compare()