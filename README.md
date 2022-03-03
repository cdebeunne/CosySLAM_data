# CosySLAM data

All you need to deal with CosyPose and CosySLAM:

-> Extrinsic calibration between mocap and camera frame with Cosypose or AprilTag

-> Produce the polynomial error model

-> Initialize IMU orientation wrt inertial frame

-> Align and compare SLAM trajectories with groundtruth

-> Handle gepetto viewer to plot dynamic trajectories

Dependencies:
---
These are the python libraries you'll need to install to run everything
* Pinocchio
* Sklearn
* OpenCV
* AprilTag
* Pandas
* Matplotlib
* Scipy 

You will also need to source ros, and a workspace with wolf_ros_objectslam if you want to generate rosbags for object slam

