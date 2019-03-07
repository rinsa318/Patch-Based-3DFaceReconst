"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2017-07-01 05:29:38
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-07 21:02:43
 ----------------------------------------------------

  Usage:
   python main.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  <path to input image>
   argvs[2]  :  <dlib_predictor_path> --> .dat file
   argvs[3]  :  patch_size   -->  defult == 10
   argvs[4]  :  overlap_size   -->  defult == 5



"""

print(__doc__)

import numpy as np
import sys
import os
import scipy.sparse
import scipy.sparse.linalg
import dlib_utils as utils
import cv2
import dlib

## my funcs
import normal2depth as nd
import obj_functions as ob
import colortransfer as ct
import patch_func as pa
import dataset_func as da





def get_mask(image, predictor_path):

  ### apply face landmark detection
  detector = dlib.get_frontal_face_detector() #Face detector
  predictor = dlib.shape_predictor(predictor_path) #Landmark identifier. Set the filename to whatever you named the downloaded file
  # image = resize_image(image, 1.0) # --> in order to, speed up
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  landmark = utils.get_landmarks(clahe.apply(gray), detector, predictor)
  # visualize_landmark = utils.visualize_landmark(image, landmark)
  landmark4mask = utils.edit_landmark(landmark)
    # cv2.imwrite("{0}/{1}_visualized_landmark.png".format(output_path, filename), visualize_landmark)


  ### create mask from fp
  mask = utils.create_mask(image, landmark4mask)
  # mesh, tri_mesh = utils.draw_each_triangles(image, landmark4mask, facet_array, "./triangle_color.npy")
  # cv2.imwrite("{0}/{1}_mask.png".format(output_path, filename), mask)

  return mask, landmark4mask





## on going
def normalize_color(image, mask, data_array, mask_array):

  ### data should be have 1 channel
  edit_data = np.zeros((data_array.shape[0], data_array.shape[1], data_array.shape[2]), dtype=np.uint8)
  

  for i in range(data_array.shape[0]):

    ### 3 channel
    transfered_image = ct.colortransfer(image, mask, data_array[i], mask_array[i])
    transfered_image[mask_array[i]!=255] = [0, 0, 0] if transfered_image.ndim == 3 else 0

    ### convert to 1 channel
    edit_data[i]  = cv2.cvtColor(transfered_image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("input", cv2.cvtColor(data_array[i], cv2.COLOR_BGR2GRAY))
    cv2.imshow("trans", edit_data[i])
    cv2.waitKey(0)

  return edit_data




def normalize_rot_scale(image, fp, data_array, fp_array):


  ### find angle and scale
  input_eye_distance = fp[42] - fp[39]
  input_length = np.linalg.norm(input_eye_distance)
  input_normalize = input_eye_distance / input_length
  angle = []
  scale = []

  for i in range(data_array.shape[0]):
    data_eye_distance = fp_array[i][42] - fp_array[i][39]
    data_length = np.linalg.norm(data_eye_distance)
    data_normalize = data_eye_distance / data_length
    scale_temp = data_length / input_length
    angle_temp = np.arccos(np.dot(data_normalize, input_normalize))
    angle.append( angle_temp )
    scale.append( scale_temp )



  ### define rot_angle, and scale_factor
  angle = np.array(angle, dtype=np.float32)
  scale = np.array(scale, dtype=np.float32)
  rot_angle = (np.mean(angle) / np.pi) * 180.0
  scale_factor = np.mean(scale)


  ### roteta and zoom in/out for normalize
  size = tuple([image.shape[1], image.shape[0]])
  center = tuple([int(size[0]/2), int(size[1]/2)])
  rotation_matrix = cv2.getRotationMatrix2D(center, -1*rot_angle, scale_factor)
  img_rot = cv2.warpAffine(image, rotation_matrix, size, flags=cv2.INTER_CUBIC)


  ### crop image or re-zoom image
  if(scale_factor < 1.0):
    y_up = center[1] - (image.shape[0] / 2.0 * scale_factor)
    y_bottom = center[1] + (image.shape[0] / 2.0 * scale_factor)
    x_up = center[0] - (image.shape[1] / 2.0 * scale_factor)
    x_bottom = center[0] + (image.shape[1] / 2.0 * scale_factor)
    dst = img_rot[int(y_up):int(y_bottom), int(x_up):int(x_bottom)]
    return dst

  else:
    y_up = center[1] - (image.shape[0] / 2.0 / scale_factor)
    y_bottom = center[1] + (image.shape[0] / 2.0 / scale_factor)
    x_up = center[0] - (image.shape[1] / 2.0 / scale_factor)
    x_bottom = center[0] + (image.shape[1] / 2.0 / scale_factor)    
    dst = img_rot[int(y_up):int(y_bottom), int(x_up):int(x_bottom)]
    return dst








def make_albedo(depth):

  albedo = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
  albedo[:, :, 0] = 0.5
  albedo[:, :, 1] = 0.5
  albedo[:, :, 2] = 0.5


  return albedo





def mask2tiny(mask, window):

  '''
  naive approach to remove noise around border
  '''

  # mask
  mask = np.array(mask, dtype=np.uint8)
  eroded = cv2.erode(mask, np.ones((int(window), int(window)), np.uint8)) # 0~1

  return eroded




def heatmap(input):
  ''' Returns a RGB heatmap
  :input: gray scale image --> numpy array
  :return: cv2 image array 
  '''
  min = np.amin(input)
  max = np.amax(input)
  rescaled = 255 * ( (input - min ) / (max - min) )

  return cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)




def normalize(normal, mask):

  n = np.zeros(normal.shape)

  for i in range(normal.shape[0]):
    for j in range(normal.shape[1]):
      
      if(mask[i][j] == 0):
        continue
        
      norm = np.linalg.norm(normal[i][j])
      n[i][j] = normal[i][j] / norm

  return n




def export_argv(outpath, argv):

  with open(outpath, "w" ) as f:
    f.write("Executed command\n")
    f.write("-->\n")
    f.write("\n")

    for i in range(len(argv)):
      f.write("{} ".format(str(argv[i])))



def main():


  #################
  ## 1. set config and load data
  #################

  ### config
  input_path = argvs[1]
  predictor_path = argvs[2]
  output_path, filename_ext = os.path.split(input_path)
  filename, ext = os.path.splitext(filename_ext)
  export_argv("{0}/{1}_executed_command.txt".format(output_path, filename), argvs)
  

  ### load data
  facet_array = np.load("./facest_list.npy")
  triangle_color_array = np.load("./triangle_color.npy")
  rgb_data = np.array(da.load_database("./data/man/rgb_data.txt", bool(0)) / 255.0, dtype=np.float32)
  rgb_data4ColorNormalize = np.array(da.load_database("./data/man/rgb_data.txt", bool(1)), dtype=np.uint8)
  normal_data = np.array(da.load_database("./data/man/normal_data.txt", bool(1)) / 255.0, dtype=np.float32)
  mask_data = da.load_database("./data/man/mask_data.txt", bool(0))
  fp_data = da.load_fp_database("./data/man/fp_data.txt")
  print("rgb data size: {}".format(rgb_data.shape))
  print("normal data size: {}".format(normal_data.shape))
  print("fp data size: {}".format(fp_data.shape))



  #################
  ## 2. load input and make mask(before normalize)
  #################
  init_image = cv2.imread(input_path, 1)
  init_mask, init_landmark = get_mask(init_image, predictor_path)
  utils.export_landmark(init_landmark, "{0}/{1}_init_landmark.txt".format(output_path, filename))
  cv2.imwrite("{0}/{1}_init_mask.png".format(output_path, filename), init_mask)



  #################
  ## 3. normalize input image to fit databaset, and calculate weight
  #################

  ### normalize rot zoom
  image_rot_zoom_normalized = normalize_rot_scale(init_image, init_landmark, rgb_data, fp_data)
  mask, landmark = get_mask(image_rot_zoom_normalized, predictor_path)
  utils.export_landmark(landmark, "{0}/{1}_landmark.txt".format(output_path, filename))
  cv2.imwrite("{0}/{1}_mask.png".format(output_path, filename), mask)
  cv2.imwrite("{0}/{1}_rot_zoom_normalized.png".format(output_path, filename), image_rot_zoom_normalized)
  
  ### normalized color
  image_normalized = ct.color_normalize(rgb_data4ColorNormalize, mask_data, image_rot_zoom_normalized, mask)
  input_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2GRAY) / 255.0
  cv2.imwrite("{0}/{1}_normalized_input.png".format(output_path, filename), image_normalized)

  ### calculate_weight
  center_image, center_list, nbrs_list = utils.kd_tree(image_normalized, landmark, facet_array, 20)
  mesh, tri_mesh = utils.draw_each_triangles(image_normalized, landmark, facet_array, triangle_color_array)
  weight = utils.calculate_weight(tri_mesh, landmark, facet_array, mask, nbrs_list)
  np.save("{0}/{1}_weight.npy".format(output_path, filename), weight)
  # cv2.imwrite("{0}/{1}_center_location.png".format(output_path, filename), center_image)
  # cv2.imwrite("{0}/{1}_weight.png".format(output_path, filename), np.array(weight[:, :, :3]*255, dtype=np.uint8))
  






  ###########################
  # 4. apply patch tiling for estimation
  ###########################

  ### config for output
  output_rgbimage = np.zeros(input_image.shape, dtype=np.float32)
  temp_rgbimage = np.zeros(input_image.shape, dtype=np.float32)
  output_normalimage = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.float32)
  temp_normalimage = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.float32)
  
  ### apply patch tiling
  dis, pos, patch_pos = pa.find_optimal_patch(input_image, weight, facet_array, rgb_data, normal_data, fp_data, patch_size, overlap_size, mask, temp_rgbimage, temp_normalimage)
  print("distance array {0}: ".format(dis.shape))
  print("input-patch position array {0}: ".format(pos.shape))
  print("data-patch position array shape {0}: ".format(patch_pos.shape))

  ### remake mask for border noise
  mask = mask2tiny(mask, 5)
  print("reconstructing image from optimal patch .........")
  
  ### reconst bgr result
  bgr_result = pa.reproduce_image(output_rgbimage, rgb_data, dis, pos, patch_pos, patch_size)
  bgr_result[mask==0] = [0, 0, 0] if bgr_result.ndim == 3 else 0
  # cv2.imwrite("{0}/{1}_estimated_rgb.png".format(output_path, filename), np.array((bgr_result*255), dtype=np.uint8))
  # print("save reconst rgb image --> {0}/{1}_estimated_rgb.png".format(output_path, filename))

  ### reconst normal result
  normal_result = pa.reproduce_image(output_normalimage, normal_data, dis, pos, patch_pos, patch_size)
  normal_result[mask==0] = [0.0, 0.0, 0.0] if normal_result.ndim == 3 else 0.0
  np.save("{0}/{1}_estimated_normal.npy".format(output_path, filename), normal_result*2.0 -1.0)
  # cv2.imwrite("{0}/{1}_estimated_normal.png".format(output_path, filename), np.array((normal_result*255), dtype=np.uint8))
  # print("save reconst normal image --> {0}/{1}_estimated_normal.png".format(output_path, filename))
  print("done!")







  ###########################
  # 5. Surface reconstruction from normal map
  ###########################
  
  ### convert bgr to normal[-1 ~ 1]
  normal = (normal_result * 2.0) - 1.0
  normal[mask == 0] = [0.0, 0.0, 0.0]
  normal = normal[:, :, ::-1] # --> (optional) depends on your coordinate system
  normal[:, :, 1] = normal[:, :, 1] * -1 # --> (optional) depends on your coordinate system

  ### normalize
  n = normalize(normal, mask)

  ### estimate depth
  depth = nd.comp_depth_4edge(mask, n)

  ### load albedo or create favarite albedo
  # cv2.imread('''albedp.png'''')
  # albedo = make_albedo(depth)
  albedo = image_rot_zoom_normalized.copy() / 255.0

  ### convert depth to ver and tri
  ver, tri = ob.Depth2VerTri(depth, mask)
  ob.writeobj("{0}/{1}_estimated_depth.obj".format(output_path, filename), ver, tri)
  ob.save_as_ply("{0}/{1}_estimated_depth.ply".format(output_path, filename), depth, n, albedo, mask, tri)
  print("save reconst result as 3d file!!")




  ###########################
  # 6. exports 2d outputs
  ###########################  
  ### create depth map
  depth_image = np.array((1.0 - (depth / np.max(depth))) * 255, dtype=np.uint8 )
  depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)

  ### combine each result
  bgr_result = cv2.cvtColor(np.array(bgr_result*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
  normal_result = np.array(normal_result*255, dtype=np.uint8)
  results = np.hstack( ( image_rot_zoom_normalized, image_normalized) )
  results = np.hstack( ( results, bgr_result) )
  results = np.hstack( ( results, normal_result) )
  results = np.hstack( ( results, depth_image) )
  
  ### save
  cv2.imwrite("{0}/{1}_outputs.png".format(output_path, filename), np.array(results, dtype=np.uint8))
  print("save result!!")




#--------------------------
#main 
#--------------------------


argvs = sys.argv
# input_fullpath, failfile_name = os.path.split(argvs[1])
# filename, extention = os.path.splitext(os.path.basename(argvs[1]))
facet_array = np.load("facest_list.npy")
if(len(argvs) > 3):
  patch_size = int(argvs[3])
  overlap_size = int(argvs[4])
else:
  patch_size = 10
  overlap_size = 5

main()