"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-04 16:52:33
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-19 14:12:22
 ----------------------------------------------------


"""


import numpy as np
import cv2
import sys


def find_correspond_point(weight, facet_array, fp_data):
  """
  calculate corresponding point between input patch and data pach
  --> estimated by input point barycentric weight

  Ex) 
  input_point: (x, y)
  Triangles around input_point: (p_1, p_2, p_3)
  weight: (w_1, w_2, w_3)
  
  (x, y) = ( (p_1 * w_1) + (p_2 * w_2) + (p_3 * w_3) )

  """

  corres_point = np.zeros((2,), dtype=np.int32)

  face_num = int(weight[3])
  tri = facet_array[face_num]

  a = fp_data[tri[0]]
  b = fp_data[tri[1]]
  c = fp_data[tri[2]]

  # be careful
  # fp_data[0] --> x, fp_data[1] --> y
  # However, you have to prepare as [0] is y, [1] is x in my code
  corres_point[1] =  (int)(weight[2]*a[0] + weight[0]*b[0] + weight[1]*c[0])
  corres_point[0] =  (int)(weight[2]*a[1] + weight[0]*b[1] + weight[1]*c[1])
  
  return corres_point




def calculate_distance(input_img, data_array):
  """
  This finction need same size of image array.
  Ex) 
  input_img = 100 * 100 
  --> each data image has to prepare 100 * 100

  """

  distance = []

  for i in range(data_array.shape[0]):

    dif = input_img - data_array[i]
    difdif = dif*dif
    distance.append(difdif.sum())

  return np.array(distance, dtype=np.float32)






def calculate_distance_all(input, input_left, input_top, input_normal_left, input_normal_top, data, data_left, data_top, data_normal_left, data_normal_top, w1=1.0, w2=0.1, w3=0.1):
  """
  This finction need same size of image array.
  Ex) 
  input_img = 100 * 100 
  --> each data image has to prepare 100 * 100


  global, left_over, top_over

  """

  distance_global = []
  distance_left = []
  distance_top = []
  distance_normal_left = []
  distance_normal_top = []

  # global
  for i in range(data.shape[0]):

    dif = input - data[i]
    difdif = dif*dif
    distance_global.append(w1*difdif.sum())

  dis_global = np.array(distance_global, dtype=np.float32)
  # print(dis_global.shape)


  # rgb left
  for i in range(data_left.shape[0]):

    dif = input_left - data_left[i]
    difdif = dif*dif
    distance_left.append(w2*difdif.sum())

  dis_left = np.array(distance_left, dtype=np.float32)
  # print(dis_left.shape)


  # rgb top
  for i in range(data_top.shape[0]):

    dif = input_top - data_top[i]
    difdif = dif*dif
    distance_top.append(w3*difdif.sum())

  dis_top = np.array(distance_top, dtype=np.float32)
  # print(dis_top.shape)


  # normal left
  for i in range(data_normal_left.shape[0]):

    dif = input_normal_left - data_normal_left[i]
    difdif = dif*dif
    distance_normal_left.append(w2*difdif.sum())

  dis_normal_left = np.array(distance_normal_left, dtype=np.float32)
  # print(dis_left.shape)


  # normal top
  for i in range(data_normal_top.shape[0]):

    dif = input_normal_top - data_normal_top[i]
    difdif = dif*dif
    distance_normal_top.append(w3*difdif.sum())

  dis_normal_top = np.array(distance_normal_top, dtype=np.float32)
  # print(dis_normal_top.shape)



  return dis_global + dis_top + dis_left + dis_normal_top + dis_normal_left





def find_optimal_patch(input_img, weight, facet_array, data_array, normal_data_array, fp_data, patch_size, overlap_size, mask, temp_rgbimage, temp_normalimage):
  """
  i = y, j = x
  only calculate global distance
  

  """

  print("start finding optimal patch .....")

  distance = [] # each patch rgb value error
  position = [] # each patch position(at the upper left)
  position_patch = [] # each patch position(at the upper left)


  for i in range(0, input_img.shape[0], patch_size - overlap_size):

    # # show progress
    h = input_img.shape[0] / (patch_size - overlap_size)
    w = i/(patch_size - overlap_size) + 1
    # print(str(i/(patch_size - overlap_size) + 1) + " / " + str(h))
    # per = float(w) / float(h) * 100
    # sys.stdout.write("\r%d" % per + " [%]")
    # sys.stdout.flush()
    progress_bar(w, h)

    if(i+patch_size > input_img.shape[0] or i > input_img.shape[0]):
      continue

    for j in range(0, input_img.shape[1], patch_size - overlap_size):

      if(j+patch_size > input_img.shape[1] or j > input_img.shape[1]):
        continue

      if(mask[i][j] == 0 and mask[i+patch_size-1][j+patch_size-1] == 0 and mask[i+patch_size-1][j] == 0 and mask[i][j+patch_size-1] == 0):
        continue

      # prepare input patch
      position.append(np.array([i, j], dtype=np.int64))
      patch_input = input_img[i:i+patch_size, j:j+patch_size]
      patch_overleft = temp_rgbimage[i:i+patch_size, j:j+overlap_size]
      patch_overtop = temp_rgbimage[i:i+overlap_size, j:j+patch_size]
      patch_normal_overleft = temp_normalimage[i:i+patch_size, j:j+overlap_size]
      patch_normal_overtop = temp_normalimage[i:i+overlap_size, j:j+patch_size]
      # print("input" + str(np.array([i, j], dtype=np.int64)))
      
      # prepare database's patchs
      w = weight[i][j]
      patch_list = []
      patch_list_left = []
      patch_list_top = []
      patch_list_normal_left = []
      patch_list_normal_top = []
      position_patch_list = []
      for k in range(data_array.shape[0]):
        # current patch's position from input weight
        point = find_correspond_point(w, facet_array, fp_data[k])

        if(point[0] < data_array[k].shape[0] and point[1] < data_array[k].shape[1] and point[0]+patch_size < data_array[k].shape[0] and point[1]+patch_size < data_array[k].shape[1] ):
          # rgb
          current_patch = data_array[k][point[0]:point[0]+patch_size, point[1]:point[1]+patch_size]
          current_patch_overleft = data_array[k][point[0]:point[0]+patch_size, point[1]:point[1]+overlap_size]
          current_patch_overtop = data_array[k][point[0]:point[0]+overlap_size, point[1]:point[1]+patch_size]
          patch_list.append(current_patch)
          patch_list_left.append(current_patch_overleft)
          patch_list_top.append(current_patch_overtop)

          # normal
          current_patch_normal_overleft = normal_data_array[k][point[0]:point[0]+patch_size, point[1]:point[1]+overlap_size]
          current_patch_normal_overtop = normal_data_array[k][point[0]:point[0]+overlap_size, point[1]:point[1]+patch_size]
          patch_list_normal_left.append(current_patch_normal_overleft)
          patch_list_normal_top.append(current_patch_normal_overtop)      

          # position
          position_patch_list.append( point )

        # error process
        # if it cannot extract patch because of current patch's position 
        # Ex) image size(100, 100), patch's left-top point is (102, 110)
        # --> use huge value array. It caused large number of distance. Thus, that patch will not select. 
        else:
          # rgb
          current_patch = np.ones((patch_size, patch_size), dtype=np.float32)
          current_patch_overtop = np.ones((overlap_size, patch_size), dtype=np.float32)
          current_patch_overleft = np.ones((patch_size, overlap_size), dtype=np.float32)
          patch_list.append(current_patch*1000)
          patch_list_top.append(current_patch_overtop*1000)
          patch_list_left.append(current_patch_overleft*1000)

          # normal
          current_patch_normal_overtop = np.ones((overlap_size, patch_size, 3), dtype=np.float32)
          current_patch_normal_overleft = np.ones((patch_size, overlap_size, 3), dtype=np.float32)
          patch_list_normal_top.append(current_patch_normal_overtop*1000)         
          patch_list_normal_left.append(current_patch_normal_overleft*1000)

          # posiiton
          position_patch_list.append([0, 0])
          continue

      patch_nparray = np.array(patch_list, dtype=np.float32)
      patch_nparray_left = np.array(patch_list_left, dtype=np.float32)
      # print(patch_nparray_left.shape)
      patch_nparray_top = np.array(patch_list_top, dtype=np.float32)
      patch_nparray_normal_left = np.array(patch_list_normal_left, dtype=np.float32)
      # print(patch_nparray_normal_left.shape)
      patch_nparray_normal_top = np.array(patch_list_normal_top, dtype=np.float32)
      position_patch.append( np.array(position_patch_list, dtype=np.int64))
      distance.append(calculate_distance_all(patch_input, patch_overleft, patch_overtop, patch_normal_overleft, patch_normal_overtop, patch_nparray, patch_nparray_left, patch_nparray_top, patch_nparray_normal_left, patch_nparray_normal_top))
      temp_rgbimage = reproduce_temp_image(temp_rgbimage, data_array, calculate_distance(patch_input, patch_nparray), np.array([i, j], dtype=np.int64), np.array(position_patch_list, dtype=np.int64), patch_size)
      temp_normalimage = reproduce_temp_image(temp_normalimage, normal_data_array, calculate_distance(patch_input, patch_nparray), np.array([i, j], dtype=np.int64), np.array(position_patch_list, dtype=np.int64), patch_size)



  print("\n")
  return np.array(distance, dtype=np.float32), np.array(position, dtype=np.float32), np.array(position_patch, dtype=np.int64)




def reproduce_image(output_image, data_array, distance_array, position_array, patch_position_array, patch_size):


  for k in range(distance_array.shape[0]):

    progress_bar(k, distance_array.shape[0] - 1)

    # set postion for input
    ip_upper_left = np.array((position_array[k][0], position_array[k][1]), dtype=np.int64)
    ip_bottom_right = np.array((position_array[k][0]+patch_size, position_array[k][1]+patch_size), dtype=np.int64)


    # find closest patch
    min_number = np.argmin(distance_array[k])


    # set position for cosest path
    dp_upper_left = np.array((patch_position_array[k][min_number][0], patch_position_array[k][min_number][1]), dtype=np.int64)
    dp_bottom_right = np.array((patch_position_array[k][min_number][0]+patch_size, patch_position_array[k][min_number][1]+patch_size), dtype=np.int64)


    # set for new image
    outpatch = output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]]
    datapatch = data_array[min_number][dp_upper_left[0]:dp_bottom_right[0], dp_upper_left[1]:dp_bottom_right[1]]


    if(outpatch.ndim < 3):

      for i in range(outpatch.shape[0]):
        for j in range(outpatch.shape[1]):
          
          if(outpatch[i][j] != 0 and datapatch[i][j] != 0):
            outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
          
          else:
            if(outpatch[i][j] == 0 and datapatch[i][j] != 0):
              outpatch[i][j] = datapatch[i][j]

            if(datapatch[i][j] == 0 and outpatch[i][j] != 0):
              continue

          output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch
    

    else:

      for i in range(outpatch.shape[0]):
        for j in range(outpatch.shape[1]):
          
          if(outpatch[i][j][0] != 0 and outpatch[i][j][1] != 0 and outpatch[i][j][2] != 0 and datapatch[i][j][0] != 0 and datapatch[i][j][1] != 0 and datapatch[i][j][2] != 0):
            outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
          
          else:
            if(outpatch[i][j][0] == 0 and outpatch[i][j][1] == 0 and outpatch[i][j][2] ==0 or datapatch[i][j][0] != 0 or datapatch[i][j][1] != 0 or datapatch[i][j][2] != 0):
              outpatch[i][j] = datapatch[i][j]
            
            if(datapatch[i][j][0] == 0 and datapatch[i][j][1] == 0 and datapatch[i][j][2] ==0 or outpatch[i][j][0] != 0 or outpatch[i][j][1] != 0 or outpatch[i][j][2] != 0):
              continue
          
          output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch

  print("")

  return output_image




def reproduce_temp_image(output_image, data_array, distance_array, position_array, patch_position_array, patch_size):


  # set postion for input
  ip_upper_left = np.array((position_array[0], position_array[1]), dtype=np.int64)
  ip_bottom_right = np.array((position_array[0]+patch_size, position_array[1]+patch_size), dtype=np.int64)


  # find closest patch
  min_number = np.argmin(distance_array)


  # set position for cosest path
  dp_upper_left = np.array((patch_position_array[min_number][0], patch_position_array[min_number][1]), dtype=np.int64)
  dp_bottom_right = np.array((patch_position_array[min_number][0]+patch_size, patch_position_array[min_number][1]+patch_size), dtype=np.int64)


  # set for new image
  outpatch = output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]]
  datapatch = data_array[min_number][dp_upper_left[0]:dp_bottom_right[0], dp_upper_left[1]:dp_bottom_right[1]]


  if(outpatch.ndim < 3):

    for i in range(outpatch.shape[0]):
      for j in range(outpatch.shape[1]):
        
        if(outpatch[i][j] != 0 and datapatch[i][j] != 0):
          outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
        
        else:
          if(outpatch[i][j] == 0):
            outpatch[i][j] = datapatch[i][j]

          if(datapatch[i][j] == 0):
            continue

        output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch
  


  else:

    for i in range(outpatch.shape[0]):
      for j in range(outpatch.shape[1]):
        
        if(outpatch[i][j][0] != 0 and outpatch[i][j][1] != 0 and outpatch[i][j][2] != 0 and datapatch[i][j][0] != 0 and datapatch[i][j][1] != 0 and datapatch[i][j][2] != 0):
          outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
        
        else:
          if(outpatch[i][j][0] == 0 and outpatch[i][j][1] == 0 and outpatch[i][j][2] ==0):
            outpatch[i][j] = datapatch[i][j]
          
          if(datapatch[i][j][0] == 0 and datapatch[i][j][1] == 0 and datapatch[i][j][2] ==0):
            continue
        
        output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch


  return output_image



def progress_bar(n, N):

  '''
  print current progress
  '''

  percent = float(n) / float(N) * 100

  ## convert percent to bar
  current = "#" * int(percent//2)
  # current = "=" * int(percent//2)
  remain = " " * int(100/2-int(percent//2))
  bar = "|{}{}|".format(current, remain)# + "#" * int(percent//2) + " " * int(100/2-int(percent//2)) + "|"
  print("\r{}:{:3.0f}[%]".format(bar, percent), end="", flush=True)