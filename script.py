import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# cap = cv.VideoCapture('en-1.mp4')
# list_img = []
# while cap.isOpened():
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # RGB_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     cv.imshow('frame', frame)
#     list_img.append(frame)
#     if cv.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv.destroyAllWindows()
#
#
# print(list_img)

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
video_path = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/output0917_mimic(smooth_lmks)_3_13.mp4'
save_path = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/gpt_demo_output0917/real_frame/'
import cv2
vidcap = cv2.VideoCapture(video_path)
# vidcap.set(cv2.CAP_PROP_FPS, 25)
success,image = vidcap.read()
cv2.imwrite(save_path+"frame%d.png" % 0, image)
count = 1
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite(save_path+"frame%d.png" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1
  print(count)



