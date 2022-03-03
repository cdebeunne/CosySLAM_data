# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
# from joblib import load
import pinocchio as pin
import rosbag
import rospy
from std_msgs.msg import Float64, String, Time, Header
from geometry_msgs.msg import Pose, Point, Quaternion
import wolf_ros_objectslam.msg as msg_cp
import utils.posemath as pm

from scipy.spatial.transform import Rotation as R

"""
Produces the rosbag that contains both imu and cosyobject messages
for cosyslam with IMU

It requires the pickle with cosypose results and the initial rosbag

arg 1 : alias

"""

if __name__ == '__main__':

    alias = sys.argv[1]
    data_path = 'data/'

    # processing of the data frame
    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    c_M_b_cosy = [pin.SE3(T) for T in df_cosypose['pose']]
    cosy_score = df_cosypose['detection_score'].values
    cosy_object = df_cosypose['object_name'].values
    cosy_ts = df_cosypose['timestamp'].values

    bag = rosbag.Bag(f'{alias}_wolf.bag', 'w')

    # Writing IMU message if available
    enable_imu = True
    if enable_imu:
        exp_path = data_path + f'{alias}.bag'
        exp_bag = rosbag.Bag(exp_path, "r")
        counter = 0

        for topic, msg, t in exp_bag.read_messages(topics=['/camera/imu']):
            t_imu = msg.header.stamp
            bag.write('/imu', msg, t_imu)

        exp_bag.close()

    dt = 1/30
    t_cur = cosy_ts[0] 
    cosy_array = []
    for i in range(len(cosy_score)):
        t = cosy_ts[i] 
        
        if (t == t_cur):
            header = Header()
            ts = rospy.Time.from_sec(t)
            header.stamp = ts
            header.seq = i
            header.frame_id = str(i)

            pose_msg = pm.isometry_to_transform(c_M_b_cosy[i])
            
            cp_msg = msg_cp.CosyObject()
            cp_msg.name = cosy_object[i]
            cp_msg.pose = pose_msg
            cp_msg.score = cosy_score[i]

            cosy_array.append(cp_msg)
        else:
            cp_arr_msg = msg_cp.CosyObjectArray()
            cp_arr_msg.header = header
            cp_arr_msg.objects = cosy_array
            bag.write('/cosypose', cp_arr_msg, ts)

            cosy_array = []
            header = Header()
            ts = rospy.Time.from_sec(t)
            header.stamp = ts
            header.seq = i
            header.frame_id = str(i)
            pose_msg = pm.isometry_to_transform(c_M_b_cosy[i])
            
            cp_msg = msg_cp.CosyObject()
            cp_msg.name = cosy_object[i]
            cp_msg.pose = pose_msg
            cp_msg.score = cosy_score[i]
            t_cur = t

            cosy_array.append(cp_msg)
    
    bag.close()
