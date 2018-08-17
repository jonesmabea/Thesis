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

## VoxelNet Model 
- [ ] Obtain baseline results 
- [ ] Implement CometML 
- [ ] Evaluate original alpha and beta values
- [ ] Evaluate original alpha and beta values with SGD

  ### Parameter Tuning 
  - [ ] Early stopping 
  - [ ] Explore kernel initialisation 
  - [ ] Implement different RPN architectures 
  
## Urban vs Non Urban Classifier 

  ### Segmentation 
   #### CityScapes Data
   - [ ] Download Data 
   - [ ] Preprocess 
  
   #### DeepLab3 Model 
   - [ ] Implement CometML 
   - [ ] Train 
  
  
   
  

## 
