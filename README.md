# Car Driving Without Cameras 

## A research project in fulfillment of MSc. Advanced Computing at the University of Bristol 

# Introduction 

Given the increased interest in autonomous vehicles, various companies are rushing forward to bring forward level 5 autonomous vehicles. 
In order to do so, these vehicles have to have an array of sensors to percieve the surrounding environment such as cameras, radar, LiDAR and so forth. 
This project, will  focus on LiDAR which projects the surrounding environment as point clouds. Specifically, the aim is to investigate the performance of LiDAR only object detection methods in urban vs non-urban contexts. 

# Progress 

## KITTI DATASET 
- [X] Download 
- [X] Separate into Train, Test, Val set 
- [X] Preprocess point clouds  
  
## Urban vs Non Urban Context Detection 
  ### Preprocessing 
   - [x] Visually classify subset of KITTI dataset to be used in training and testing. 
   
  
  ### Image Context 
   - [x] Run DeepLab V3 Model on KITTI Dataset
   - [x] Obtain semantic histograms
   - [x] Train image context classifier using semantic histograms
  
  ### Point Cloud Context 
   - [x] Explore classification models suitable to point clouds
   - [x] Train pointcloud context classifier using features
  
  
  ## Models(VoxelNet and AVOD) 

  ### VoxelNet
  - [X] Obtain baseline results with pretrained model
  - [X] Evaluate original alpha and beta values
  - [X] Evaluate original alpha and beta values with SGD
  - [X] Notebooks for interactive training and testing
  - [X] Implement utility tools and model functions for validation dataset.
  - [X] Implement GPU Monitor for inference code. 
  #### Parameter Tuning 
  - [ ] Early stopping 
  - [X] Explore kernel initialisation 
  - [X] Implement different RPN architectures for pedestrian and cyclist models.
  - [X] Implement focal loss function.

   ### AVOD 
  - [X] Obtain baseline results with pretrained model
  - [X] Notebooks for interactive training and testing
  - [X] Implement GPU Monitor for inference code. 

    
   ### Analysis
  - [x] Research GPU metrics to be logged during model inference 
  - [X] Notebooks for interactive analysis 

   
    

