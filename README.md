# Patch-Based 3D Face Reconstruction

This is my undergraguate project, and this is following a paper below.  

"3D Facial Geometry Reconstruction using Patch Database",  
Tsukasa Nozawa, Takuya Kato, Pavel A. Savkin, Naoki Nozawa, Shigeo Morishima,  
SIGGRAPH Poster 2016  
[[Paper](https://dl.acm.org/citation.cfm?doid=2945078.2945102 "Paper")]  

<p align="center">
  <img src="./temp/figure1.png" width=90%>
</p>
  
## Environment
Ubuntu 18.04  
Python3.6(Anaconda3-5.2.0)



## Dependency

+ OpenCV3
+ Dlib

+ [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "")





## How to run


### Dataset

To run my code, you have to prepare dataset below.  
68 landmark points and mask image can extract by using creat_data.py

+ RGB images, Normal map images and Mask images
+ taken under the same measurement environment(lighting environment)
+ facial position, scale and rotation have to normalize  
  --> We normalized them by fitting both inner corners of eyes at the same point between all image
+ 68 landmark points for each face is needed for take correscponding --> dlib






### Running the tests

```
python main.py argvs[1] argvs[2] argvs[3] argvs[4]

argvs[1] : input image path 
argvs[2] : Landmark identifier −> shape predictor 68 face landmarks.dat 
argvs[3] : (optinal) patch size −> default = 10[pixel]
argvs[4] : (optinal) overlap size −> default = 5[pixel]

```


