# -*- coding: utf-8 -*-

from sqlite3 import Timestamp
import sys

sys.path.insert(0, "/home/cdebeunne/catkin_ws/src")
import numpy as np
import pandas as pd
import pinocchio as pin
import rosbag
import rospy
from std_msgs.msg import Float64, String, Time, Header
from geometry_msgs.msg import Pose, Point, Quaternion
import wolf_ros_objectslam.msg as msg_cp
import utils.posemath as pm
from scipy.spatial.transform import Rotation as R

"""
A script to add the timestamp from the rosbag to cosypose's detection data stored in
pkl files
takes results_scenario.pkl and returns results_scenario_ts.pkl

arg 1 : alias 
"""

if __name__ == '__main__':

    alias = sys.argv[1]
    data_path = 'data/'

    # processing of the data frame
    df_cosypose = pd.read_pickle(data_path+f'results_{alias}.pkl')
    print(df_cosypose)
    bag_images = rosbag.Bag(data_path+f'{alias}.bag', "r")
    timestamps_df = []

    counter = 1
    for _, msg, t in bag_images.read_messages(topics=['/camera/color/image_raw']):
        df_frame = df_cosypose.loc[df_cosypose["frame_id"] == counter]
        for k in range(len(df_frame)):
            timestamps_df.append(msg.header.stamp.to_sec())
        # print(msg.header.stamp.to_sec())
        # print('vs')
        # print(t.to_sec())
        counter += 1
        
    df_cosypose["timestamp"] = timestamps_df
    print(df_cosypose)
    df_cosypose.to_pickle(data_path+f'results_{alias}_ts.pkl', protocol=2)
