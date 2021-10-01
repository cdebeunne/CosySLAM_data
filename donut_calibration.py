#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import degrees
import numpy as np
import pinocchio as pin
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.wrapper import MocapWrapper


class CostFrameCalibration:
    def __init__(self, bm_M_cm_traj, c_M_b_cosy_traj, cm_M_c):
        self.x_arr = []
        self.cost_arr = []
        self.cm_M_c = cm_M_c
        self.bm_M_cm_traj = bm_M_cm_traj
        self.c_M_b_cosy_traj = c_M_b_cosy_traj
        self.N = len(bm_M_cm_traj)
        self.N_res = 3*N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        bm_M_b = pin.exp6(x[:6])  # currently estimated transfo as SE3 Lie group
        b_M_bm = bm_M_b.inverse()
        res = np.zeros(self.N_res)
        for i in range(self.N):
            bm_M_cm = self.bm_M_cm_traj[i]
            c_M_k  = self.c_M_b_cosy_traj[i]

            # Let's set the transformation around z axis
            r_k = R.from_euler('z', x[6+i], degrees=True)
            R_k = r_k.as_matrix()
            T_k = np.identity(4)
            T_k[0:3,0:3] = R_k
            k_M_b = pin.SE3(T_k)

            res[3*i:3*i+3]   = (c_M_k * k_M_b * b_M_bm * bm_M_cm * self.cm_M_c).translation
            # res[6*i+3:6*i+6] = pin.log3((c_M_k * k_M_b * b_M_bm * bm_M_cm * self.cm_M_c).rotation)
        
        self.cost_arr.append(np.linalg.norm(res))

        return res

class NewCostFrameCalibration:
    def __init__(self, bm_M_cm_traj, c_M_b_cosy_traj, cm_M_c):
        self.x_arr = []
        self.cost_arr = []
        self.cm_M_c = cm_M_c
        self.bm_M_cm_traj = bm_M_cm_traj
        self.c_M_b_cosy_traj = c_M_b_cosy_traj
        self.N = len(bm_M_cm_traj)
        self.N_res = 6*N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        bm_M_b = pin.exp6(x[:6])  # currently estimated transfo as SE3 Lie group
        c_M_cm = pin.exp6(x[6:])
        res = np.zeros(self.N_res)
        for i in range(self.N):
            bm_M_cm = self.bm_M_cm_traj[i]
            c_M_b  = self.c_M_b_cosy_traj[i]
            c_M_b_mocap = c_M_cm * bm_M_cm.inverse() * bm_M_b
            z = np.array([0,0,1])

            res[4*i] = np.linalg.norm(c_M_b.translation) - np.linalg.norm(c_M_b_mocap.translation)
            res[4*i+1:4*i+4] = np.cross(c_M_b.rotation @ z.T, c_M_b_mocap.rotation @ z.T)
        
        self.cost_arr.append(np.linalg.norm(res))

        return res

if __name__ == '__main__':

    alias = 'calib_donut'
    data_path = 'data/'

    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    df_gt = pd.read_pickle(data_path + f'groundtruth_{alias}.pkl')
    mocap_wrapper = MocapWrapper(df_gt)

    # let's get the good calibration for the camera frame
    calib = np.load(data_path+'calibration_switchBIS.npz')
    cm_M_c = pin.SE3(calib['cm_M_c'])

    # cosypose trajectory
    counter = 0
    c_M_b_traj = [pin.SE3(T[0]) for T in df_cosypose['pose']]
    N = len(c_M_b_traj)
    bm_M_cm_traj,_ = mocap_wrapper.trajectory_generation(df_cosypose)

    # filter
    c_M_b_traj = c_M_b_traj
    bm_M_cm_traj = bm_M_cm_traj
    N = len(c_M_b_traj)

    # cost function
    cost = NewCostFrameCalibration(bm_M_cm_traj, c_M_b_traj, cm_M_c)
    
    x0 = np.zeros(12)  
    for i in range(2):
        # r = optimize.least_squares(cost.f, x0, jac='3-point', method='trf', verbose=2)
        r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)
        x0 = r.x
    nu_b = r.x[:6]
    nu_c = r.x[6:]

    # recover the transformation with exp
    c_M_cm_est = pin.exp6(nu_c) 
    bm_M_b_est = pin.exp6(nu_b)
    print('c_M_cm_est\n', c_M_cm_est)
    print('bm_M_b_est\n', bm_M_b_est)

    np.savez(data_path+'calibration_donut.npz', c_M_cm=c_M_cm_est, bm_M_b=bm_M_b_est)
    
    print('r.cost')
    print(r.cost)
    print()

    # residuals RMSE
    res = r.fun.reshape((N, 6))
    rmse_arr = np.sqrt(np.mean(res**2, axis=0))
    print(rmse_arr)

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
