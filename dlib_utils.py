"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: n1.n2n3__n4n5@ruri.waseda.jp
  @Date: 2017-07-18 23:14:59
  @Last Modified by:   wazano318
  @Last Modified time: 2017-11-14 15:18:22
 ----------------------------------------------------

  Usage:
   python dlib_utils.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  input file path
   argvs[2]  :  Landmark identifier   -->   .dat


"""


# print(__doc__)


import numpy as np
import cv2
import dlib
import sys
import random
import os
# from scipy.interpolate import interp1d
# from scipy.interpolate import splprep, splev
# from  scipy.interpolate  import  interp1d
from PIL import Image
from sklearn.neighbors import NearestNeighbors



def get_landmarks(image, detector, predictor):

  print("start get landmark ........")
  detections = detector(image, 1)
  for k,d in enumerate(detections): #For all detected face instances individually
  	shape = predictor(image, d) #Draw Facial Landmarks with the predictor class

  # create new Landmark numpy arry
  landmarks = np.empty([shape.num_parts, 2], dtype = int)
  for i in range(shape.num_parts):
    landmarks[i][0] = shape.part(i).x
    landmarks[i][1] = shape.part(i).y

  if len(detections) > 0:
    print("done!!")
    return landmarks
  else: #If no faces are detected, return error message to other function to handle
    print("error")
    landmarks_error = ["error"]
    return landmarks_error



def visualize_landmark(image, landmarks, input_fullpath, filename):

  output= image.copy()
  if len(landmarks) == 68:

    # create new vector about each parts
    outline = np.empty([17, 2], dtype = int)
    r_eyebrows = np.empty([5, 2], dtype = int)
    l_eyebrows = np.empty([5, 2], dtype = int)
    nose_up = np.empty([4, 2], dtype = int)
    nose_down = np.empty([6, 2], dtype = int)
    r_eye = np.empty([6, 2], dtype = int)
    l_eye = np.empty([6, 2], dtype = int)
    mouth_out = np.empty([12, 2], dtype = int)
    mouth_in = np.empty([8, 2], dtype = int)

    for i in range(len(landmarks)):

      if i < 17:
        outline[i] = landmarks[i]
      elif 17 <= i & i < 22:
        r_eyebrows[i-17] = landmarks[i]
      elif 22 <= i & i < 27:
        l_eyebrows[i-22] = landmarks[i]
      elif 27 <= i & i < 31:
        nose_up[i-27] = landmarks[i]
        if i == 30:
          nose_down[i-30] = landmarks[i]
      elif 31 <= i & i < 36:
        nose_down[i-30] = landmarks[i]                
      elif 36 <= i & i < 42:
        r_eye[i-36] = landmarks[i]
      elif 42 <= i & i < 48:
        l_eye[i-42] = landmarks[i]
      elif 48 <= i & i < 60:
        mouth_out[i-48] = landmarks[i]
      else:
        mouth_in[i-60] = landmarks[i]

  # print(nose_up[-1], nose_down[0], landmarks[30])

    # draw line
    outline = outline.reshape((-1, 1, 2))
    r_eyebrows = r_eyebrows.reshape((-1, 1, 2))
    l_eyebrows = l_eyebrows.reshape((-1, 1, 2))
    nose_up = nose_up.reshape((-1, 1, 2))
    nose_down = nose_down.reshape((-1, 1, 2))
    r_eye = r_eye.reshape((-1, 1, 2))
    l_eye = l_eye.reshape((-1, 1, 2))
    mouth_out = mouth_out.reshape((-1, 1, 2))
    mouth_in = mouth_in.reshape((-1, 1, 2))

    cv2.polylines(output, [outline], False, (0,255,0), 1)
    cv2.polylines(output, [r_eyebrows], False, (0,255,0), 1)
    cv2.polylines(output, [l_eyebrows], False, (0,255,0), 1)
    cv2.polylines(output, [nose_up], False, (0,255,0), 1)
    cv2.polylines(output, [nose_down], True, (0,255,0), 1)
    cv2.polylines(output, [r_eye], True, (0,255,0), 1)
    cv2.polylines(output, [l_eye], True, (0,255,0), 1)
    cv2.polylines(output, [mouth_out], True, (0,255,0), 1)
    cv2.polylines(output, [mouth_in], True, (0,255,0), 1)

    # draw point and number
    for i in range(len(landmarks)):
      pos = (landmarks[i][0], landmarks[i][1])
      cv2.circle(output, pos, 1, (0,0,255), thickness=2)
      cv2.putText(output, str(i+1), pos, cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite(input_fullpath + "/" + filename + "_landmark.png", output)
    # cv2.imshow("Landmark detection result", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("save landmark image --> landmark_image.png")
    # return output


  else:
    print('error!!')
    # return output


def create_mask_image(image, landmarks, input_fullpath, filename):
  """
  this function is need 68 landmark

  """
  
  output = image.copy()
  black_img = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)

  base = landmarks[27] - landmarks[8]
  # initial_base = base / 6
  initial_base = 1.5 * (base / 6)
  
  outline = []
  for i in range(17):
    outline.append(landmarks[i])

  outline.append(landmarks[26] + initial_base)
  outline.append(landmarks[24] + initial_base)
  outline.append(landmarks[19] + initial_base)
  outline.append(landmarks[17] + initial_base)
  outline.append(landmarks[0])


  outline = np.array(outline, dtype=int)
  outline = outline.reshape((-1, 1, 2))
  cv2.fillPoly(black_img, pts =[outline], color=255)
  cv2.imwrite(input_fullpath + "/" + filename + "_mask.png", black_img)
  # cv2.imshow("Landmark detection result", output)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  print("save landmark outline image --> mask_image.png")

  for y in range(black_img.shape[0]):
    for x in range(black_img.shape[1]):

      if(black_img[y][x] > 240):
        black_img[y][x] = True
      else:
        black_img[y][x] = False

  return black_img



def create_mask_image_ongoing(image, landmarks, input_fullpath, filename):

  """
  create mask by using b-spline curve
  not yet finish 

  """


  interporation_number = 100
  output = image.copy()
  black_img = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)

  base = landmarks[27] - landmarks[8]
  initial_base = base / 6


  # facial bottom outline
  x_b = np.empty((15,), dtype=np.float32)
  y_b = np.empty((15,), dtype=np.float32)
  for i in range(15):
    x_b[i] = landmarks[i+1][0]
    y_b[i] = landmarks[i+1][1]


  # apply interporation
  f2_b = interp1d( x_b, y_b, kind='cubic' )
  x_bn = np.linspace( np.min(x_b), np.max(x_b), interporation_number )
  y_bn = f2_b(x_bn)


  # facial top outline
  x_t = np.empty((6,), dtype=np.float32)
  y_t = np.empty((6,), dtype=np.float32)

  x_t[0] = landmarks[1][0] # + initial_base[0]
  y_t[0] = landmarks[1][1] # + initial_base[1]
  x_t[1] = landmarks[18][0] + initial_base[0]/ 1.5
  y_t[1] = landmarks[18][1] + initial_base[1]/ 1.5
  x_t[2] = landmarks[19][0] + initial_base[0]
  y_t[2] = landmarks[19][1] + initial_base[1]
  x_t[3] = landmarks[24][0] + initial_base[0]
  y_t[3] = landmarks[24][1] + initial_base[1]
  x_t[4] = landmarks[25][0] + initial_base[0]/ 1.5
  y_t[4] = landmarks[25][1] + initial_base[1]/ 1.5
  x_t[5] = landmarks[15][0] # + initial_base[0]
  y_t[5] = landmarks[15][1] # + initial_base[1]

  # apply interporation
  f2_t = interp1d( x_t, y_t, kind='cubic' )
  x_tn = np.linspace( np.min(x_t), np.max(x_t), interporation_number )
  y_tn = f2_t(x_tn)


  # create final outline array
  outline = np.empty((interporation_number*2, 2), dtype=int)
  for i in range(interporation_number*2):
    if(i >= interporation_number):
      num = i - interporation_number + 1 
      outline[i][0] = x_tn[-num]
      outline[i][1] = y_tn[-num]
    else:
      outline[i][0] = x_bn[i]
      outline[i][1] = y_bn[i]


  # draw mask image by using outline
  outline = outline.reshape((-1, 1, 2))
  cv2.fillPoly(black_img, pts =[outline], color=255)
  cv2.imwrite(input_fullpath + "/" + filename + "_mask_image_test.png", black_img)
  print("save landmark outline image --> mask_image.png")
  # return black_img


def resize_image(image, rate):
    
  new_hight = image.shape[0]*rate
  new_width = image.shape[1]*rate
  resize_image = cv2.resize(image, ((int)(new_width), (int)(new_hight)))

  return resize_image



def landmark_output(landmarks, input_fullpath, filename):
  """
  this function can export landmarks as text file

  """
 
  # # to export pure future point
  f = open(input_fullpath + "/" + filename + "_landmark.txt", "w")

  for i in range(len(landmarks)):
    f.write(str(landmarks[i][0]) + "," + str(landmarks[i][1]) + "\n")


  # # to export modified version
  new_landmarks = landmarks.copy()
  base = landmarks[27] - landmarks[8]
  # initial_base = base / 6 + ((base / 6) / 6)
  initial_base = 1.5 * (base / 6)

  for i in range(new_landmarks.shape[0]) :
    if(i == 17 or i == 19 or i == 24 or i == 26):
      new_landmarks[i] = landmarks[i] + initial_base

  f = open(input_fullpath + "/" + filename + "_modified_landmark.txt", "w")

  for i in range(len(new_landmarks)):
    f.write(str(new_landmarks[i][0]) + "," + str(new_landmarks[i][1]) + "\n")

  print("save landmark --> output_landmark.txt")

  return new_landmarks




def rect_contains(rect, point):
  """
  Check if a point is inside a rectangle

  """
  
  if point[0] < rect[0]:
    return False
  elif point[1] < rect[1]:
    return False
  elif point[0] > rect[2]:
    return False
  elif point[1] > rect[3]:
    return False
  return True



def create_facet_list(landmarks, pt1, pt2, pt3):
  """
  return triangle argment number list

  """

  index = []

  a = np.array([int(pt1[0]), int(pt1[1])], dtype=int)
  b = np.array([int(pt2[0]), int(pt2[1])], dtype=int)
  c = np.array([int(pt3[0]), int(pt3[1])], dtype=int)
  
  for i, fp in enumerate(landmarks):
    if(np.allclose(fp,a)):
      index.append(i)

  for j, fp in enumerate(landmarks):
    if(np.allclose(fp,b)):
      index.append(j)

  for k, fp in enumerate(landmarks):
    if(np.allclose(fp,c)):
      index.append(k)

  return np.array(index, dtype=int)






def draw_each_triangles(image, input_fullpath, filename, landmarks, facet_array, facet_color, wirecolor=(255, 255, 255)):

  output_triangle = image.copy()
  output_mesh = image.copy()

  tri_color_new = np.load(facet_color)
  # print(tri_color_new.shape)

  new_landmarks = landmarks.copy()
  base = landmarks[27] - landmarks[8]
  # initial_base = base / 6 + ((base / 6) / 6)
  initial_base = 1.5 * (base / 6)

  for i in range(new_landmarks.shape[0]) :
    if(i == 17 or i == 19 or i == 24 or i == 26):
      new_landmarks[i] = landmarks[i] + initial_base

  # tri_color = []
  for i in range(facet_array.shape[0]):

      # # for mesh
      A = (new_landmarks[facet_array[i][0]][0], new_landmarks[facet_array[i][0]][1])
      B = (new_landmarks[facet_array[i][1]][0], new_landmarks[facet_array[i][1]][1])
      C = (new_landmarks[facet_array[i][2]][0], new_landmarks[facet_array[i][2]][1])

      cv2.line(output_mesh, A, B, wirecolor, 2, 0)
      cv2.line(output_mesh, B, C, wirecolor, 2, 0)
      cv2.line(output_mesh, C, A, wirecolor, 2, 0)

      # # for triangles
      tri = []
      tri.append(new_landmarks[facet_array[i]][0])
      tri.append(new_landmarks[facet_array[i]][1])
      tri.append(new_landmarks[facet_array[i]][2])
      tri = np.array(tri, dtype=int)

      center_point = tri.mean(axis = 0)
      center = (int(center_point[0]), int(center_point[1]))


      color = (tri_color_new[i][0], tri_color_new[i][1], tri_color_new[i][2])
      cv2.fillPoly(output_triangle, [tri], color)
      # cv2.putText(output_triangle, str(i+1), center, cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255), 1, cv2.LINE_AA)


  cv2.imwrite(input_fullpath + "/" + filename + "_mesh.png", output_mesh)
  cv2.imwrite(input_fullpath + "/" + filename + "_triangle.png", output_triangle)
  print("drew each triangles and mesh")


  return output_mesh, output_triangle



def kd_tree(image, input_fullpath, filename,  landmarks, facet_array, num):

  """
  firnd nearest face by using kd-tree

  1. calcu each triangles center point
  2. apply kd-tree
  3. export nearest face index

  """

  print("stert kd-tree .......")

  output = image.copy()



  # initilize fp list
  new_landmarks = landmarks.copy()
  base = landmarks[27] - landmarks[8]
  # initial_base = base / 6
  initial_base = 1.5 * (base / 6)

  for i in range(new_landmarks.shape[0]) :
    if(i == 17 or i == 19 or i == 24 or i == 26):
      new_landmarks[i] = landmarks[i] + initial_base


  # # 1. create center point list
  center_list = np.zeros((facet_array.shape[0], 2), dtype=int)
  center_list = []
  for i in range(facet_array.shape[0]):

      A = (new_landmarks[facet_array[i][0]][0], new_landmarks[facet_array[i][0]][1])
      B = (new_landmarks[facet_array[i][1]][0], new_landmarks[facet_array[i][1]][1])
      C = (new_landmarks[facet_array[i][2]][0], new_landmarks[facet_array[i][2]][1])
      mean_p = (new_landmarks[facet_array[i][0]] + new_landmarks[facet_array[i][1]] + new_landmarks[facet_array[i][2]]) / 3

      cv2.line(output, A, B, (255, 255, 255), 2, 0)
      cv2.line(output, B, C, (255, 255, 255), 2, 0)
      cv2.line(output, C, A, (255, 255, 255), 2, 0)

      pos = (mean_p[0], mean_p[1])
      cv2.circle(output, pos, 1, (0,0,255), thickness=2)
      center_list.append( mean_p )

  cv2.imwrite(input_fullpath + "/" + filename + "_center_location.png", output)
  center_list = np.array(center_list, dtype=int)



  # # 2. apply kd-tree
  all_pixel = [] 
  for y in range(output.shape[0]):
    for x in range(output.shape[1]):
      point = [x, y]
      all_pixel.append(point)

  all_pixel = np.array(all_pixel, dtype=int)

  nbrs = NearestNeighbors(n_neighbors=num, algorithm='kd_tree').fit(np.array(center_list, dtype=int))
  dis, ind = nbrs.kneighbors(all_pixel)


  # # 3.export nearest face index
  nbrs_list = np.zeros((image.shape[0], image.shape[1], num), dtype=int)
  for y in range(nbrs_list.shape[0]):
    for x in range(nbrs_list.shape[1]):

      nbrs_list[y][x] = ind[nbrs_list.shape[1]*y + x]

  print("done!!")
  return center_list, nbrs_list




def calculate_weight(image, input_fullpath, filename, landmarks, facet_array, mask, nbrs_list):
  """
  calculate each point weight 

  1. find triangles that is incuding point (x, y) from nearest face(kd-tree)
  2. calculate barycentric weight
  3. export nearest face index

  """

  print("start calculate each point weight  ......")

  # initilaize landmark
  new_landmarks = landmarks.copy()
  base = landmarks[27] - landmarks[8]
  # initial_base = base / 6
  initial_base = 1.5 * (base / 6)

  for i in range(new_landmarks.shape[0]) :
    if(i == 17 or i == 19 or i == 24 or i == 26):
      new_landmarks[i] = landmarks[i] + initial_base


  # prepare array to export weight
  output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
  weight = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)

  for y in range(output.shape[0]):

    h = output.shape[0]
    per = float(y+1) / float(h) * 100
    sys.stdout.write("\r%d" % per + " [%]")
    sys.stdout.flush()

    for x in range(output.shape[1]):


      # apply mask
      if(mask[y][x]==1):

        flag = 1
        for i in range(nbrs_list.shape[2]):

          # if finish finding triangles, ignore other nearest_face
          if(flag == 1):
            # find triangle that includs (y, x)
            nearest_face_num = nbrs_list[y][x][i]
            a = np.array([new_landmarks[facet_array[nearest_face_num][0]][1], new_landmarks[facet_array[nearest_face_num][0]][0], 0], dtype=np.float32)
            b = np.array([new_landmarks[facet_array[nearest_face_num][1]][1], new_landmarks[facet_array[nearest_face_num][1]][0], 0], dtype=np.float32)
            c = np.array([new_landmarks[facet_array[nearest_face_num][2]][1], new_landmarks[facet_array[nearest_face_num][2]][0], 0], dtype=np.float32)

            p = np.array([y, x, 0], dtype=np.float32)

            ap = a - p
            bp = b - p
            cp = c - p

            ca = a - c
            ab = b - a
            bc = c - b

            cross_1 = np.cross(ca, cp)
            cross_2 = np.cross(ab, ap)
            cross_3 = np.cross(bc, bp)

            # cross_1_init = cross_1 / np.linalg.norm(cross_1)
            # cross_2_init = cross_2 / np.linalg.norm(cross_2)
            # cross_3_init = cross_3 / np.linalg.norm(cross_3)

            # if find triangle, calculate weight
            if(cross_1[2] <= 0 and cross_2[2] <= 0 and cross_3[2] <= 0):
              P = np.array([int(y), int(x)])
              w = barycentric_weight(new_landmarks, facet_array[nearest_face_num], P)
              output[y][x] = (w[0], w[1], w[2])
              weight[y][x] = (w[0], w[1], w[2], nearest_face_num)
              flag = 0


            elif(cross_1[2] >= 0 and cross_2[2] >= 0 and cross_3[2] >= 0 and flag):
              P = np.array([int(y), int(x)])
              w = barycentric_weight(new_landmarks, facet_array[nearest_face_num], P)
              output[y][x] = (w[0], w[1], w[2])
              weight[y][x] = (w[0], w[1], w[2], nearest_face_num)
              flag = 0

          else:
            continue

      else:
        continue

  pil_output = Image.fromarray((output*255).astype(np.uint8))
  pil_output.save(input_fullpath + "/" + filename + "_weight.png")
  np.save(input_fullpath + "/" + filename + "_weight_array.npy", weight)

  print(" ")
  print("done!!")
  return weight




def barycentric_weight(landmarks, triangles, P):

  A = np.array([int(landmarks[triangles[0]][1]), int(landmarks[triangles[0]][0])])
  B = np.array([int(landmarks[triangles[1]][1]), int(landmarks[triangles[1]][0])])
  C = np.array([int(landmarks[triangles[2]][1]), int(landmarks[triangles[2]][0])])


  # area of ABC
  AB = B - A
  AC = C - A
  ABC_area = np.linalg.norm(np.cross(AB, AC))

  # area of ABP
  AP = P - A
  ABP_area = np.linalg.norm(np.cross(AB, AP))
  
  # area of BCP
  BC = C - B
  BP = P - B
  BCP_area = np.linalg.norm(np.cross(BC, BP))
  
  # area of CAP
  CP = P - C
  CAP_area = np.linalg.norm(np.cross(-AC, CP))

  # P = ux + vy + wz
  # u = CAP_area / all, v = ABP_area / all, w = BCP_area / all
  # weight = np.array([CAP_area / ABC_area, ABP_area / ABC_area, BCP_area / ABC_area], dtype=np.float32)
  weight = np.array([CAP_area / ABC_area, ABP_area / ABC_area, BCP_area / ABC_area])


  return weight





# ================================
# Set up some required objects
# ================================

def main():
  # load facet list(from landmark point)
  # np.save("facest_array_fixed.npy", facet_array)
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
  landmark = get_landmarks(clahe_image, detector, predictor)
  visualize_landmark(image, landmark, input_fullpath, filename)
  landmark_edit = landmark_output(landmark, input_fullpath, filename)


  # create mask from fp and 
  mask = create_mask_image(image, landmark, input_fullpath, filename)


  # export
  mesh, tri_mesh = draw_each_triangles(image, input_fullpath, filename, landmark, facet_array, "./triangle_color.npy")


  # find nearest face
  center_list, nbrs_list = kd_tree(image, input_fullpath, filename, landmark, facet_array, 18)


  # calculate_weight
  weight = calculate_weight(tri_mesh, input_fullpath, filename, landmark, facet_array, mask, nbrs_list)



# main()





