"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2018-06-19 16:09:45
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-19 14:17:48
 ----------------------------------------------------

  Usage:
   python colortransfer.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  src image   -->   style
   argvs[2]  :  target image

"""



import numpy as np
import cv2
import sys
import os


def image_stats(image, mask):

  # compute the mean and standard deviation of each channel
  (l, a, b) = cv2.split(image)
  (lMean, lStd) = (l[mask!=0].mean(), l[mask!=0].std())
  (aMean, aStd) = (a[mask!=0].mean(), a[mask!=0].std())
  (bMean, bStd) = (b[mask!=0].mean(), b[mask!=0].std())
 
  return (lMean, lStd, aMean, aStd, bMean, bStd)



def image_array_stats(image_array, mask_array):

  lMean_data = 0.0
  lStd_data = 0.0
  aMean_data = 0.0
  aStd_data = 0.0
  bMean_data = 0.0
  bStd_data = 0.0

  num = float(image_array.shape[0])

  for i in range(image_array.shape[0]):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image_array[i])
    (lMean, lStd) = (l[mask_array[i]!=0].mean(), l[mask_array[i]!=0].std())
    (aMean, aStd) = (a[mask_array[i]!=0].mean(), a[mask_array[i]!=0].std())
    (bMean, bStd) = (b[mask_array[i]!=0].mean(), b[mask_array[i]!=0].std())


    lMean_data += lMean
    lStd_data += lStd
    aMean_data += aMean
    aStd_data += aStd
    bMean_data += bMean
    bStd_data += bStd

  return (lMean_data/num, lStd_data/num, aMean_data/num, aStd_data/num, bMean_data/num, bStd_data/num)


def scale_array(arr):

  # NumPy array that has been scaled to be in [0, 255] range
  scaled = np.clip(arr, 0, 255)
  # else:
  #   scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
  #   scaled = _min_max_scale(arr, new_range=scale_range)

  return scaled



def colortransfer(src, src_mask, tar, tar_mask):


  source = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype("float32")
  target = cv2.cvtColor(tar, cv2.COLOR_BGR2LAB).astype("float32")




  ## ------------------------------
  # 2. calculate std each channel
  ## ------------------------------
  # compute the mean and standard deviation of each channel
  (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source, src_mask)
  (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target, tar_mask)



  ## ------------------------------
  # 3. apply color transfer
  ## ------------------------------
  # print("start color transfer")

  ### subtract the means from the target image
  (l, a, b) = cv2.split(target)
  l -= lMeanTar
  a -= aMeanTar
  b -= bMeanTar

  ### scale by the standard deviations
  l = (lStdTar / (lStdSrc + 1e-8)) * l
  a = (aStdTar / (aStdSrc + 1e-8)) * a
  b = (bStdTar / (bStdSrc + 1e-8)) * b

  ### add in the source mean
  l += lMeanSrc
  a += aMeanSrc
  b += bMeanSrc

  ### have to scale array 0 to 255
  l = scale_array(l)
  a = scale_array(a)
  b = scale_array(b)

  ### merge the channels together and convert back to the RGB color
  ### space, being sure to utilize the 8-bit unsigned integer data type
  if((aMeanSrc == 128.0 and bMeanSrc == 128.0) or (aMeanTar == 128.0 and bMeanTar == 128.0)):
    ag = np.ones((tar.shape[0], tar.shape[1]), dtype=np.float32) * 128.0
    bg = np.ones((tar.shape[0], tar.shape[1]), dtype=np.float32) * 128.0
    transfer = cv2.merge([l, ag, bg])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
  
  else:
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    # print("-------> done!!!!")
    # print("")
    # print("")
  return transfer



def color_normalize(data_array, mask_array, image, mask):

  '''
  normalize input image color to database images mean
  --> using simple color transfer

  '''


  ## convert to bgr to lab
  ## source
  src_array = data_array.copy()
  for i in range(data_array.shape[0]):
    src_array[i] = cv2.cvtColor(data_array[i], cv2.COLOR_BGR2LAB).astype("float32")

  ## target
  target = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")




  ## ------------------------------
  # 2. calculate std each channel
  ## ------------------------------
  # compute the mean and standard deviation of each channel
  (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_array_stats(src_array, mask_array)
  (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target, mask)


  ## ------------------------------
  # 3. apply color transfer
  ## ------------------------------
  # print("start color transfer")
    
  ### subtract the means from the target image
  (l, a, b) = cv2.split(target)
  l -= lMeanTar
  a -= aMeanTar
  b -= bMeanTar

  ### scale by the standard deviations
  l = (lStdTar / (lStdSrc + 1e-8)) * l
  a = (aStdTar / (aStdSrc + 1e-8)) * a
  b = (bStdTar / (bStdSrc + 1e-8)) * b

  ### add in the source mean
  l += lMeanSrc
  a += aMeanSrc
  b += bMeanSrc

  ### have to scale array 0 to 255
  l = scale_array(l)
  a = scale_array(a)
  b = scale_array(b)

  ### merge the channels together and convert back to the RGB color
  ### space, being sure to utilize the 8-bit unsigned integer data type
  if((aMeanSrc == 128.0 and bMeanSrc == 128.0) or (aMeanTar == 128.0 and bMeanTar == 128.0)):
    ag = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 128.0
    bg = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 128.0
    transfer = cv2.merge([l, ag, bg])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

  else:
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)    


  return transfer





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



def main():


  ## ------------------------------
  ## load arg
  ## ------------------------------
  argvs = sys.argv
  data = argvs[1]
  mask = argvs[2]
  tar_path = argvs[3]
  tar_mask = argvs[4]



  ## ------------------------------
  ## difine filename
  ## ------------------------------
  # filename_src, ext_src = os.path.splitext( os.path.basename(src_path) )
  filename_tar, ext_tar = os.path.splitext( os.path.basename(tar_path) )
  # path_src, filefullname_src = os.path.split( tar_path )
  path_tar, filefullname_tar = os.path.split( tar_path )
  output_path = "{0}/{1}_color_normalized{2}".format(path_tar, filename_tar, ext_tar)

  # check name
  # print("src --> " + filefullname_src)
  # print("tar --> " + filefullname_tar)




  ## ------------------------------
  ## load image
  ## ------------------------------
  data_array = np.array(load_database(data, bool(1)), dtype=np.uint8)
  mask_array = np.array(load_database(mask, bool(0)), dtype=np.uint8)
  tar = cv2.imread(tar_path, 1)
  mask = cv2.imread(tar_mask, 0)




  ## ------------------------------
  ## apply color transfer
  ## ------------------------------
  transfered = colortransfer(data_array[0], mask_array[0], tar, mask ) 
  # transfered = colortransfer(tar, mask, data_array[0], mask_array[0]) 
  # transfered = color_normalize(data_array, mask_array, tar, mask)




  # ## ------------------------------
  # # show and save result
  # ## ------------------------------
  # print("save output as --> {}".format(output_path))
  cv2.imshow(output_path, transfered)
  # cv2.imwrite(output_path, transfered)
  cv2.waitKey(0)




# main()


