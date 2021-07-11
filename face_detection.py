#################### MTCNN #################
!pip install mtcnn
import os
import numpy as np
from PIL import Image
import glob
import re
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import mtcnn
from mtcnn import MTCNN

fmodel = MTCNN()


X = []
left_eye = []
right_eye = []
nose = []
lip = []


j = 0
d = 0
real = []
attack = []
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
dirlist = sorted_alphanumeric(glob.glob('/content/Test_files/*.avi'))

for i in range(len(dirlist)):
  if (dirlist[i][-5]=='1'):
    real.append(dirlist[i])
  else:
    attack.append(dirlist[i])
qw = 0
for dd in (attack):
  print('#########',d)
  print('########################',100*qw/len(attack))
  qw+=1
  d = 0
  cap = cv2.VideoCapture(dd)
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if ((j%10)==0):
      a = fmodel.detect_faces(frame)
      if (a==[]):
        continue
      x1, y1, width, height = a[0]['box']
      if (width<100):
        continue
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height
      face_pixels = frame[y1:y2, x1:x2]
      image = Image.fromarray(face_pixels)
      image = image.resize((200,260))
      face_array = np.asarray(image)
      X.append(face_array)
      r_e_x,r_e_y = a[0]['keypoints']['right_eye']
      if ((r_e_x < 55) or (r_e_y < 55) ):
        continue
      ex_patch = frame[r_e_y-52:r_e_y+52,r_e_x-52:r_e_x+52]
      image = Image.fromarray(ex_patch)
      image = image.resize((64,64))
      patch_array = np.asarray(image)
      right_eye.append(patch_array)
      l_e_x,l_e_y = a[0]['keypoints']['left_eye']
      if ((l_e_x < 55)or (l_e_y < 55)):
        continue
      ex_patch = frame[l_e_y-52:l_e_y+52,l_e_x-52:l_e_x+52]
      image = Image.fromarray(ex_patch)
      image = image.resize((64,64))
      patch_array = np.asarray(image)
      left_eye.append(patch_array)
      n_x,n_y = a[0]['keypoints']['nose']
      if ((n_x < 55)or (n_y < 55)):
        continue
      ex_patch = frame[n_y-52:n_y+52,n_x-52:n_x+52]
      image = Image.fromarray(ex_patch)
      image = image.resize((64,64))
      patch_array = np.asarray(image)
      nose.append(patch_array)
      l_m_x,l_m_y = a[0]['keypoints']['mouth_left']
      r_m_x,r_m_y = a[0]['keypoints']['mouth_right']
      if ((l_m_x < 55)or (l_m_y < 55)):
        continue
      ex_patch = frame[l_m_y-45:l_m_y+65,l_m_x:r_m_x]
      ex_patch = Image.fromarray(ex_patch)
      ex_patch = ex_patch.resize((64,64))
      patch_array = np.asarray(ex_patch)
      lip.append(patch_array)
      d+=1
    j+=1
X = np.array(X)
right_eye = np.array(right_eye)
left_eye = np.array(left_eye)
nose = np.array(nose)
lip = np.array(lip)
