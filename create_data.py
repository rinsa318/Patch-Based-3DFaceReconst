"""
 ----------------------------------------------------
  @Author: wazano318
  @Affiliation: Waseda University
  @Email: n1.n2n3__n4n5@ruri.waseda.jp
  @Date: 2017-09-11 01:22:33
  @Last Modified by:   wazano318
  @Last Modified time: 2017-11-14 05:44:28
 ----------------------------------------------------

  Usage:
   python create_data.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  input file path
   argvs[2]  :  Landmark identifier   -->   .dat


"""

print(__doc__)

import dlib_utils as utils
import numpy as np
import sys
import os
import cv2
import dlib



facet_array = np.load("facest_list.npy")


argvs = sys.argv
input_fullpath, tailfile_name = os.path.split(argvs[1])
filename, extention = os.path.splitext(os.path.basename(argvs[1]))

image = cv2.imread(argvs[1])

detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor(argvs[2]) #Landmark identifier. Set the filename to whatever you named the downloaded file


# apply face landmark detection
# image = resize_image(image, 1.0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)
landmark = utils.get_landmarks(clahe_image, detector, predictor)
utils.visualize_landmark(image, landmark, input_fullpath, filename)
landmark_edit = utils.landmark_output(landmark, input_fullpath, filename)

# create mask from fp and 
mask = utils.create_mask_image(image, landmark, input_fullpath, filename)


