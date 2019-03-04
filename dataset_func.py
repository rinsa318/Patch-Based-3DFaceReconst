"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-04 16:54:33
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-05 08:06:36
 ----------------------------------------------------


"""

import numpy as np
import cv2
import sys


def load_database(data_list_path, flag):

  """
  load image from .txt file.
  That .txt file has to write each data path.
  Ex) 
  ./{path for image1}
  ./{path for image2}

  ...

  ./{path for imageN}

  flag = true --> bgr
  flag = false --> gray

  """

  data = []

  # if(flag):
  with open(data_list_path, 'r') as f:
    for line in f:
      # image_data = np.array(Image.open(line.rstrip("\n")), dtype=np.float32) / 255.0
      image_data = cv2.imread(line.rstrip("\n"), flag)# / 255.0
      data.append(image_data)

  return np.array(data, dtype=np.uint8)





def load_fp_database(data_list_path):

  """
  load fp from .txt file.
  That .txt file has to write each data path.
  Ex) 
  ./{path for fp_1}
  ./{path for fp_2}

  ...

  ./{path for fp_N}

  """

  data = []
  with open(data_list_path, 'r') as f:

    for line in f:
      data_temp = []

      for l in open(line.rstrip("\n")).readlines():
        fp_list = l[:-1].split(',')
        data_temp.append( [fp_list[0], fp_list[1]] )
      
      data.append( data_temp )

  return np.array(data, dtype=np.int32)