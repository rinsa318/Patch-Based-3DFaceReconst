"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: n1.n2n3__n4n5@ruri.waseda.jp
  @Date: 2017-07-01 05:29:38
  @Last Modified by:   wazano318
  @Last Modified time: 2017-11-14 15:21:53
 ----------------------------------------------------

  Usage:
   python main.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  <path to input image>
   argvs[2]  :  <dlib_predictor_path> --> .dat file
   argvs[3]  :  patch_size   -->  defult == 10
   argvs[4]  :  overlap_size   -->  defult == 5



"""

print(__doc__)

from PIL import Image
import matplotlib.pyplot
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import sys
import os
import numpy.linalg
import scipy.sparse
#import scipy.linsolve
import scipy.sparse.linalg
import matplotlib.pyplot
import mpl_toolkits.mplot3d.axes3d
import dlib_utils as utils
import cv2
import dlib





def load_database(data_list_path, flag):

  """
  load image from .txt file.
  That .txt file has to write each data path.
  Ex) 
  ./{path for image1}
  ./{path for image2}

  ...

  ./{path for imageN}

  flag = true --> image
  flag = false --> mask

  """

  data = []

  if(flag):
    with open(data_list_path, 'r') as f:
      for line in f:
        image_data = np.array(Image.open(line.rstrip("\n")), dtype=float) / 255.0
        data.append(image_data)

    return np.array(data, dtype=np.float32)


  else:
    with open(data_list_path, 'r') as f:
      for line in f:
        image_data = np.asarray(Image.open(line.rstrip("\n")).convert('L'), dtype=float) / 255.0
        data.append(image_data)

    return np.array(data, dtype=np.float32)






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
        point = []
        point.append( fp_list[0] )
        point.append( fp_list[1] )
        data_temp.append( point )
      data.append( data_temp )

  return np.array(data, dtype=np.int)






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

  corres_point = np.zeros((2,), dtype=int)

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

  print("start find optimal patch .....")

  distance = [] # each patch rgb value error
  position = [] # each patch position(at the upper left)
  position_patch = [] # each patch position(at the upper left)

  # num = 0
  for i in range(0, input_img.shape[0], patch_size - overlap_size):

    # show progress
    h = input_img.shape[0] / (patch_size - overlap_size)
    w = i/(patch_size - overlap_size) + 1
    # print(str(i/(patch_size - overlap_size) + 1) + " / " + str(h))
    per = float(w) / float(h) * 100
    sys.stdout.write("\r%d" % per + " [%]")
    sys.stdout.flush()

    if(i+patch_size > input_img.shape[0] or i > input_img.shape[0]):
      continue

    for j in range(0, input_img.shape[1], patch_size - overlap_size):

      if(j+patch_size > input_img.shape[1] or j > input_img.shape[1]):
        continue

      if(mask[i][j] == 0 and mask[i+patch_size-1][j+patch_size-1] == 0 and mask[i+patch_size-1][j] == 0 and mask[i][j+patch_size-1] == 0):
        continue

      # prepare input patch
      position.append(np.array([i, j], dtype=int))
      patch_input = input_img[i:i+patch_size, j:j+patch_size]
      patch_overleft = temp_rgbimage[i:i+patch_size, j:j+overlap_size]
      patch_overtop = temp_rgbimage[i:i+overlap_size, j:j+patch_size]
      patch_normal_overleft = temp_normalimage[i:i+patch_size, j:j+overlap_size]
      patch_normal_overtop = temp_normalimage[i:i+overlap_size, j:j+patch_size]
      # print("input" + str(np.array([i, j], dtype=int)))
      
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

        # error proces
        # if it cannot extract patch because of current patch's position 
        # Ex) image size(100, 100), patch's left-top point is (102, 110)
        # --> use huge value array. It caused larg number of distance. Thus, that patch will not select. 
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
      position_patch.append( np.array(position_patch_list, dtype=int))
      distance.append(calculate_distance_all(patch_input, patch_overleft, patch_overtop, patch_normal_overleft, patch_normal_overtop, patch_nparray, patch_nparray_left, patch_nparray_top, patch_nparray_normal_left, patch_nparray_normal_top))
      temp_rgbimage = reproduce_temp_image(temp_rgbimage, data_array, calculate_distance(patch_input, patch_nparray), np.array([i, j], dtype=int), np.array(position_patch_list, dtype=int), patch_size)
      temp_normalimage = reproduce_temp_image(temp_normalimage, normal_data_array, calculate_distance(patch_input, patch_nparray), np.array([i, j], dtype=int), np.array(position_patch_list, dtype=int), patch_size)
      # pil_rgb_output = Image.fromarray((temp_rgbimage*255).astype(np.uint8))
      # pil_rgb_output.save(input_fullpath + "/temp" + str(num) + "_estimated_rgb.png")
      # pil_normal_output = Image.fromarray((temp_normalimage*255).astype(np.uint8))
      # pil_normal_output.save(input_fullpath + "/temp" + str(num) + "_estimated_rgb.png")
      # num = num + 1


  print(" ")
  return np.array(distance, dtype=np.float32), np.array(position, dtype=np.float32), np.array(position_patch, dtype=int)




def reproduce_image(output_image, data_array, distance_array, position_array, patch_position_array, patch_size):


  for k in range(distance_array.shape[0]):

    # set postion for input
    ip_upper_left = np.zeros(2, dtype=int)
    ip_bottom_right = np.zeros(2, dtype=int)
    ip_upper_left[0] = position_array[k][0]
    ip_upper_left[1] = position_array[k][1]
    ip_bottom_right[0] = position_array[k][0] + patch_size
    ip_bottom_right[1] = position_array[k][1] + patch_size

    # find closest patch
    min_number = np.argmin(distance_array[k])
    # print(min_number)

    # set position for cosest path
    dp_upper_left = np.zeros(2, dtype=int)
    dp_bottom_right = np.zeros(2, dtype=int)
    dp_upper_left[0] = patch_position_array[k][min_number][0]
    dp_upper_left[1] = patch_position_array[k][min_number][1]
    dp_bottom_right[0] = patch_position_array[k][min_number][0] + patch_size
    dp_bottom_right[1] = patch_position_array[k][min_number][1] + patch_size


    outpatch = output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]]
    datapatch = data_array[min_number][dp_upper_left[0]:dp_bottom_right[0], dp_upper_left[1]:dp_bottom_right[1]]


    if(outpatch.ndim < 3):
      if(np.max(outpatch) != 0):
        for i in range(outpatch.shape[0]):
          for j in range(outpatch.shape[1]):
            if(outpatch[i][j] != 0):
              outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
            else:
              outpatch[i][j] = datapatch[i][j]
            output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch
      else:
        output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = datapatch

    else:
      if(np.max(outpatch) != 0):
        for i in range(outpatch.shape[0]):
          for j in range(outpatch.shape[1]):
            if(outpatch[i][j][0] != 0 and outpatch[i][j][1] != 0 and outpatch[i][j][2] != 0):
              outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
            else:
              outpatch[i][j] = datapatch[i][j]
            output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch
      else:
        output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = datapatch      

  return output_image




def reproduce_temp_image(output_image, data_array, distance_array, position_array, patch_position_array, patch_size):


  # set postion for input
  ip_upper_left = np.zeros(2, dtype=int)
  ip_bottom_right = np.zeros(2, dtype=int)
  ip_upper_left[0] = position_array[0]
  ip_upper_left[1] = position_array[1]
  ip_bottom_right[0] = position_array[0] + patch_size
  ip_bottom_right[1] = position_array[1] + patch_size


  # set position for cosest path
  min_number = np.argmin(distance_array)
  dp_upper_left = np.zeros(2, dtype=int)
  dp_bottom_right = np.zeros(2, dtype=int)
  dp_upper_left[0] = patch_position_array[min_number][0]
  dp_upper_left[1] = patch_position_array[min_number][1]
  dp_bottom_right[0] = patch_position_array[min_number][0] + patch_size
  dp_bottom_right[1] = patch_position_array[min_number][1] + patch_size


  outpatch = output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]]
  datapatch = data_array[min_number][dp_upper_left[0]:dp_bottom_right[0], dp_upper_left[1]:dp_bottom_right[1]]

  if(outpatch.ndim < 3):
    if(np.max(outpatch) != 0):
      for i in range(outpatch.shape[0]):
        for j in range(outpatch.shape[1]):
          if(outpatch[i][j] != 0):
            outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
          else:
            outpatch[i][j] = datapatch[i][j]
          output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch
    else:
      output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = datapatch

  else:
    if(np.max(outpatch) != 0):
      for i in range(outpatch.shape[0]):
        for j in range(outpatch.shape[1]):
          if(outpatch[i][j][0] != 0 and outpatch[i][j][1] != 0 and outpatch[i][j][2] != 0):
            outpatch[i][j] = outpatch[i][j]/2 + datapatch[i][j]/2
          else:
            outpatch[i][j] = datapatch[i][j]
          output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = outpatch
    else:
      output_image[ip_upper_left[0]:ip_bottom_right[0], ip_upper_left[1]:ip_bottom_right[1]] = datapatch      

  return output_image



def surface_reconstruction(normal_image, mask_image):
  # compute normal value from rgb value2
  normal_image = normal_image * 2.0 - 1.0
  normal_image /= np.sqrt(np.sum(normal_image ** 2, axis=2)).reshape(normal_image.shape[:2] + (1,))


  print "creating matrices"
  A = scipy.sparse.lil_matrix((normal_image.shape[0]*normal_image.shape[1]*2, normal_image.shape[0]*normal_image.shape[1]))
  #b = scipy.sparse.lil_matrix((normal_image.shape[0]*normal_image.shape[1]*2, 1))
  b = np.zeros((normal_image.shape[0]*normal_image.shape[1]*2))

  print "filling matrices"
  for y in xrange(normal_image.shape[0]-1):
      for x in xrange(normal_image.shape[1]):
          #if mask_image[y,x] > 0.9 and mask_image[y+1,x] > 0.9 and mask_image[y,x+1] > 0.9:
          #if mask_image[y,x] > 0.9:
          #if all([mask_image[y_prime, x_prime] > 0.9 for y_prime in xrange(y-1, y+2) for x_prime in xrange(x-1, x+2)]):
          pixel = y * normal_image.shape[1] + x
          #normal = normal_image[y,x,:] * 2.0 - 1.0
          #normal = normal if (math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2) - 1.0)**2 < 0.01 else [0.0, 0.0, 1.0]
          normal = normal_image[y, x, :] if mask_image[y, x] > 0.9 else np.array([0.0, 0.0, 1.0])
          
  ##        A[pixel, pixel] = normal[2]
  ##        A[pixel, pixel+1] = -normal[2]
  ##        A[pixel*2, pixel] = normal[2]
  ##        A[pixel*2, pixel+normal_image.shape[1]] = -normal[2]
  ##        b[pixel] = normal[0]
  ##        b[pixel*2] = normal[1]
          A[2*pixel, pixel] = -normal[2]
          if x<normal_image.shape[1]:
              A[2*pixel, pixel+1] = normal[2]
          b[2*pixel] = -normal[0]

          A[2*pixel+1, pixel] = -normal[2]
          if y<normal_image.shape[0]:
              A[2*pixel+1, pixel+normal_image.shape[1]] = normal[2]
          b[2*pixel+1] = -normal[1]

  print "converting"
  A = A.tocsr()
  #b = b.tocsr()

  print "computing least-squares solution"
  #x = scipy.linsolve.spsolve(A, b)
  d = scipy.sparse.linalg.lsqr(A, b)

  #d = scipy.sparse.linalg.qmr(A, b)

  depths = np.zeros((normal_image.shape[0], normal_image.shape[1], 3), dtype=np.uint8)

  #dmin = d[0].min()
  #dmax = d[0].max()
  dmin = min([d[0][y*normal_image.shape[1]+x] for y in xrange(1,normal_image.shape[0]-1) for x in xrange(1,normal_image.shape[1]-1) if mask_image[y,x] > 0.9 and mask_image[y-1,x] > 0.9 and mask_image[y+1,x] > 0.9 and mask_image[y,x-1] > 0.9 and mask_image[y,x+1] > 0.9])
  dmax = max([d[0][y*normal_image.shape[1]+x] for y in xrange(1,normal_image.shape[0]-1) for x in xrange(1,normal_image.shape[1]-1) if mask_image[y,x] > 0.9 and mask_image[y-1,x] > 0.9 and mask_image[y+1,x] > 0.9 and mask_image[y,x-1] > 0.9 and mask_image[y,x+1] > 0.9])

  for y in xrange(normal_image.shape[0]-1):
      for x in xrange(normal_image.shape[1]-1):
          #if mask_image[y,x] > 0.9 and mask_image[y+1,x] > 0.9 and mask_image[y,x+1] > 0.9:
          #if mask_image[y,x] > 0.9:
          #if all([mask_image[y_prime, x_prime] > 0.9 for y_prime in xrange(y-1, y+2) for x_prime in xrange(x-1, x+2)]):
          pixel = y * normal_image.shape[1] + x
          # depths[y,x,:] = int(255*(d[0][pixel] - dmin) / (dmax - dmin))
          depths[y,x,:] = int(255*(d[0][pixel]))


  # Image.fromarray(depths, 'RGB').save("sphere.7.depth.png")
  # np.savetxt('test.csv', d, delimiter=',')

  step = 5
  xarr = np.array([[i for i in xrange(1,normal_image.shape[1]-1,step)] for j in xrange(1,normal_image.shape[0]-1,step)])
  yarr = np.array([[j for i in xrange(1,normal_image.shape[1]-1,step)] for j in xrange(1,normal_image.shape[0]-1,step)])
  zarr = np.array([[d[0][y*normal_image.shape[1]+x]-dmin for x in xrange(1,normal_image.shape[1]-1,step)] for y in xrange(1,normal_image.shape[0]-1,step)])

  #fig = pylab.figure()
  fig = matplotlib.pyplot.figure()
  ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig)
  ax.plot_wireframe(xarr, yarr, zarr)
  # ax.set_zlim([0, 150])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Depth')

  #p = np.array([[-(normal_image[y,x,0]*2.0-1.0)/(normal_image[y,x,2]*2.0-1.0) for x in xrange(normal_image.shape[1])] for y in xrange(normal_image.shape[0])])
  #zarr2 = np.array([[sum(p[y,:x]) for x in xrange(1,normal_image.shape[1]-1,step)] for y in xrange(1,normal_image.shape[0]-1,step)])
  #fig = pylab.figure()
  #fig = matplotlib.pyplot.figure()
  #matplotlib.pyplot.title("fake")
  #ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig)
  #ax.plot_wireframe(xarr, yarr, zarr2)
  #ax.set_xlabel('X')
  #ax.set_ylabel('Y')
  #ax.set_zlabel('Depth')

  #pylab.show()
  matplotlib.pyplot.show()



def apply_mask_cut(image, mask_image):

  output = image.copy()
  # print(output.shape[0])
  # print(output.shape[1])
  # print(output.ndim)

  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      # print("y: " + str(y))
      # print("x: " + str(x))
      if(output.ndim == 3):
        output[y][x] = image[y][x][:] if mask_image[y][x] > 0.9 else np.array([0.0, 0.0, 0.0])
      else:
        output[y][x] = image[y][x] if mask_image[y][x] > 0.9 else np.array([0.0])
              
  return output



def main():

  #################
  # load database
  #################
  rgb_data = load_database("./data/man/rgb_data.txt", bool(1))
  normal_data = load_database("./data/man/normal_data.txt", bool(1))
  # mask_data = load_database("./data/man/normal_data.txt", bool(0))
  fp_data = load_fp_database("./data/man/fp_data.txt")
  print(rgb_data.shape) 
  print(normal_data.shape) 
  print(fp_data.shape)



  #################
  # prepare input
  #################
  input_image = np.array(Image.open(argvs[1]).convert("L"), dtype=float) / 255.0
  image = cv2.imread(argvs[1])
  facet_array = np.load("facest_list.npy")

  # apply face landmark detection
  detector = dlib.get_frontal_face_detector() #Face detector
  predictor = dlib.shape_predictor(argvs[2]) #Landmark identifier. Set the filename to whatever you named the downloaded file
  # image = resize_image(image, 1.0)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  clahe_image = clahe.apply(gray)
  landmark = utils.get_landmarks(clahe_image, detector, predictor)
  utils.visualize_landmark(image, landmark, input_fullpath, filename)
  landmark_edit = utils.landmark_output(landmark, input_fullpath, filename)


  # create mask from fp and 
  mask = utils.create_mask_image(image, landmark, input_fullpath, filename)
  mesh, tri_mesh = utils.draw_each_triangles(image, input_fullpath, filename, landmark, facet_array, "./triangle_color.npy")

  # calculate_weight
  center_list, nbrs_list = utils.kd_tree(image, input_fullpath, filename, landmark, facet_array, 18)
  weight = utils.calculate_weight(tri_mesh, input_fullpath, filename, landmark, facet_array, mask, nbrs_list)


 

  ###########################
  # create output image array
  ###########################
  output_rgbimage = np.zeros(input_image.shape, dtype=np.float32)
  temp_rgbimage = np.zeros(input_image.shape, dtype=np.float32)
  output_normalimage = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.float32)
  temp_normalimage = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.float32)


  ###########################
  # apply patch tiling
  ###########################
  dis, pos, patch_pos = find_optimal_patch(input_image, weight, facet_array, rgb_data, normal_data, fp_data, patch_size, overlap_size, mask, temp_rgbimage, temp_normalimage)
  print("distance array" + str(dis.shape))
  print("input-patch position array" + str(pos.shape))
  print("data-patch position array shape" + str(patch_pos.shape))
  # print(patch_pos)



  ###########################
  # export result
  ###########################
  print("reconstructing image from optimal patch .........")
  # # save rgb reesult
  rgb_result = reproduce_image(output_rgbimage, rgb_data, dis, pos, patch_pos, patch_size)
  masked_rgb_result = apply_mask_cut(rgb_result, mask)
  pil_rgb_output = Image.fromarray((masked_rgb_result*255).astype(np.uint8))
  pil_rgb_output.save(input_fullpath + "/" + filename + "_estimated_rgb.png")
  # pil_rgb_output.show()
  print("save reconst rgb image --> " + str(filename) + "_estimated_rgb.png")

  # save normal result
  normal_result = reproduce_image(output_normalimage, normal_data, dis, pos, patch_pos, patch_size)
  masked_normal_result = apply_mask_cut(normal_result, mask)
  pil_normal_output = Image.fromarray((masked_normal_result*255).astype(np.uint8))
  pil_normal_output.save(input_fullpath + "/" + filename + "_estimated_normal.png")
  # pil_normal_output.show()
  print("save reconst normal image --> " + str(filename) + "_estimated_normal.png")
  print("done!")


  ###########################
  # surface reconstruction
  ###########################
  # reconstruct surface from normal map 
  surface_reconstruction(masked_normal_result, mask)





#--------------------------
#main 
#--------------------------


argvs = sys.argv
input_fullpath, failfile_name = os.path.split(argvs[1])
filename, extention = os.path.splitext(os.path.basename(argvs[1]))
facet_array = np.load("facest_list.npy")
if(len(argvs) > 3):
  patch_size = int(argvs[3])
  overlap_size = int(argvs[4])
else:
  patch_size = 10
  overlap_size = 5

main()