import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import pinocchio as pin
from utils.wrapper import MocapWrapper, ApriltagWrapper
import cv2
import apriltag
from scipy import optimize
from scipy.spatial.transform import Rotation as R
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mocap_objects_frame_calibration import CostFrameCalibration

if __name__ == '__main__':
    alias = sys.argv[1]
    data_path = 'data/'
    file_path = data_path + '{}.bag'.format(alias)

    bag = rosbag.Bag(file_path, "r")
    df_gt = pd.read_pickle(data_path + f'groundtruth_{alias}.pkl')

    # calibration matrix and matrix of undistorted image
    mtx = np.array([[938.45928,   0.     , 642.83447],
    [0.     , 933.97625, 335.00451],
    [0.     ,   0.     ,   1.     ]])
    dist = np.array([0.104629, -0.166991, -0.004681, -0.003171, 0.000000])
    w = 1280
    h = 720
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0, (w,h))
    tag_size = 0.161

    # generate trajectories
    at_wrapper = ApriltagWrapper(bag, tag_size, mtx, dist)
    df_apriltag = at_wrapper.trajectory_generation()
    mocap_wrapper = MocapWrapper(df_gt)
    c_M_t_traj = [pin.SE3(T) for T in df_apriltag['pose']]
    tm_M_cm_traj,_ = mocap_wrapper.trajectory_generation(df_apriltag)

    # cost function
    cost = CostFrameCalibration(tm_M_cm_traj, c_M_t_traj)
    N = len(c_M_t_traj)
    x0 = np.zeros(12)  # chosen to be [nu_c, nu_b]
    for i in range(2):
        r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)
        x0 = r.x
    
    nu_c = r.x[:6]
    nu_b = r.x[6:]

    # recover the transformation with exp
    cm_M_c_est = pin.exp6(nu_c) 
    bm_M_b_est = pin.exp6(nu_b)
    print('cm_M_c_est\n', cm_M_c_est)
    print('bm_M_b_est\n', bm_M_b_est)
    np.savez(data_path+f'calibration_{alias[:-1]}.npz', cm_M_c=cm_M_c_est, bm_M_b=bm_M_b_est)
    
    print('r.cost')
    print(r.cost)
    print()

    # residuals RMSE
    res = r.fun.reshape((N, 6))
    rmse_arr = np.sqrt(np.mean(res**2, axis=0))

    # examine the problem jacobian at the solution
    J = r.jac
    H = J.T @ J
    u, s, vh = np.linalg.svd(H, full_matrices=True)

    plt.figure('cost evolution')
    plt.plot(np.arange(len(cost.cost_arr)), np.log(cost.cost_arr))
    plt.xlabel('Iterations')
    plt.ylabel('Residuals norm')

    plt.figure('Hessian singular values')
    plt.bar(np.arange(len(s)), np.log(s))
    plt.xlabel('degrees of freedom')
    plt.ylabel('log(s)')

    plt.show()





