import os
import argparse
from scipy import optimize
import sys
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import utils.posemath as pm
import pandas as pd
from utils.wrapper import SLAMWrapper
import seaborn as sns

import rosbag
import rospy

'''
A Script to compare trajectories from the results of a slam wrapped by SLAMWrapper 
'''

class MocapSLAMCalibration:
    def __init__(self, mocap_M_camera_traj, slam_M_camera_traj):
        self.x_arr = []
        self.cost_arr = []
        self.slam_M_camera_traj = slam_M_camera_traj
        self.mocap_M_camera_traj = mocap_M_camera_traj
        self.N = len(mocap_M_camera_traj)
        self.N_res = 3*self.N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        mocap_nu_slam = x[:6]  # currently estimated transfo as se3 6D vector representation
        cm_nu_camera = x[6:12]
        scale = x[12] # scale necessary for visual slam only, set to one for VI systems
        mocap_M_slam = pin.exp6(mocap_nu_slam)  # currently estimated transfo as SE3 Lie group
        cm_M_camera = pin.exp6(cm_nu_camera)


        res = np.zeros(self.N_res)
        for i in range(self.N):
            mocap_M_cm = self.mocap_M_camera_traj[i]
            slam_M_camera = self.slam_M_camera_traj[i].copy()
            slam_M_camera.translation = slam_M_camera.translation*scale
            res[3*i:3*i+3]   = (mocap_M_slam * slam_M_camera).translation - (mocap_M_cm * cm_M_camera).translation
            
        self.cost_arr.append(np.linalg.norm(res))

        return res


if __name__ == '__main__':
    bag_gt_path = sys.argv[1]
    bag_vins_path = sys.argv[2]

    
    slam_traj_list = []

    # open rosbags
    bag_mocap = rosbag.Bag(bag_gt_path, "r")
    bag_vins = rosbag.Bag(bag_vins_path, "r")

    # compute the trajectory
    vinsWrapper = SLAMWrapper(bag_vins, bag_mocap)
    mocap_M_cm_traj_complete, mocap_M_cm_traj, vins_M_camera_traj, timestamp = vinsWrapper.trajectory_generation()

    # Calibration between mocap and slam frames
    cost = MocapSLAMCalibration(mocap_M_cm_traj, vins_M_camera_traj)
    N = len(mocap_M_cm_traj)
    x0 = np.zeros(13)  # chosen to be [nu_c, nu_b]
    x0[12] = 1
    # x0 = np.zeros(12)
    r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)
    mocap_M_vins = pin.exp6(r.x[:6])
    cm_M_camera = pin.exp6(r.x[6:12])
    scale = r.x[12]

    print('transform mocap camera frame to optical frame')
    print(cm_M_camera)
    print('transform mocap fram to slam frame')
    print(mocap_M_vins)
    print('scale')
    print(scale)

    # residuals RMSE
    res = r.fun.reshape((N, 3))
    rmse_trans = np.sqrt(np.mean([err*err for err in res]))
    print('translation error')
    print(rmse_trans)

    # examine the problem jacobian at the solution
    J = r.jac
    H = J.T @ J
    u, s, vh = np.linalg.svd(H, full_matrices=True)

    # plt.figure('cost evolution')
    # plt.plot(np.arange(len(cost.cost_arr)), np.log(cost.cost_arr))
    # plt.xlabel('Iterations')
    # plt.ylabel('Residuals norm')

    # plt.figure('Hessian singular values')
    # plt.bar(np.arange(len(s)), np.log(s))
    # plt.xlabel('degrees of freedom')
    # plt.ylabel('log(s)')

    # plt.show()

    for elem in vins_M_camera_traj:
        elem.translation = elem.translation * scale

    groundtruth = np.array([(mocap_M_cm * cm_M_camera).translation for mocap_M_cm in mocap_M_cm_traj_complete])
    slam = np.array([(mocap_M_vins * vins_M_camera).translation for vins_M_camera in vins_M_camera_traj])
    slam_traj_list.append(slam)

    sns.set_theme(style="darkgrid")
    fig = plt.figure('Trajectory comparison')
    plt.plot(groundtruth[:,0], groundtruth[:,1], '.', markersize=3, label='mocap')
    plt.plot(slam_traj_list[:,0], slam_traj_list[:,1], '.',  markersize=3, label='CosySLAM $\Delta_t$=1.0s')
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')

    plt.legend()
    plt.show()

