# Patch-Based 3D Face Reconstruction

This project is following a paper below.  

"3D Facial Geometry Reconstruction using Patch Database",  
Tsukasa Nozawa, Takuya Kato, Pavel A. Savkin, Naoki Nozawa, Shigeo Morishima,  
SIGGRAPH Poster 2016  
[[Paper](http://delivery.acm.org/10.1145/2950000/2945102/a24-nozawa.pdf?ip=133.3.201.13&id=2945102&acc=ACTIVE%20SERVICE&key=D2341B890AD12BFE%2EE7C54C29BAE894E8%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1543305734_bba8c0d8e3fcca4cdadddb6d07e78b6d "Paper")]  


<img src="./temp/figure1.png" width=100%>

## Environment
Ubuntu 16.04  
Python2.7(Anaconda2-5.2.0)



## Dependency

+ OpenCV
+ Dlib
+ [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "")





## How to run


### Dataset

To run my code, you have to prepare dataset below.  
68 landmark points and mask image can extract by using creat_data.py

+ RGB images, Normal map images and Mask images
+ taken under the same measurement environment(lighting environment)
+ facial position, scale and rotation have to normalize. --> We normalize them by fitting both inner corners
of eyes at the same point between all image
+ 68 landmark points for each face is needed for take correscponding --> dlib






### Running the tests

```
python main.py argvs[1] argvs[2] argvs[3] argvs[4]

argvs[1] : input image path 
argvs[2] : Landmark identifier −> shape predictor 68 face landmarks.dat 
argvs[3] : (optinal) patch size −> default = 10[pixel]
argvs[4] : (optinal) overlap size −> default = 5[pixel]

```


