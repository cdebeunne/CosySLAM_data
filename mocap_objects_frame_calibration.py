#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from utils.wrapper import MocapWrapper


class CostFrameCalibration:
    def __init__(self, bm_M_cm_traj, c_M_b_cosy_traj):
        self.x_arr = []
        self.cost_arr = []
        self.bm_M_cm_traj = bm_M_cm_traj
        self.c_M_b_cosy_traj = c_M_b_cosy_traj
        self.N = len(bm_M_cm_traj)
        self.N_res = 6*N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        nu_c = x[:6]  # currently estimated transfo as se3 6D vector representation
        nu_b = x[6:]  # currently estimated transfo as se3 6D vector representation
        cm_M_c = pin.exp6(nu_c)  # currently estimated transfo as SE3 Lie group
        bm_M_b = pin.exp6(nu_b)  # currently estimated transfo as SE3 Lie group
        b_M_bm = bm_M_b.inverse()
        res = np.zeros(self.N_res)
        for i in range(self.N):
            bm_M_cm = self.bm_M_cm_traj[i]
            c_M_b  = self.c_M_b_cosy_traj[i]
            res[6*i:6*i+3]   = (c_M_b * b_M_bm * bm_M_cm * cm_M_c).translation*15
            res[6*i+3:6*i+6] = pin.log3((c_M_b * b_M_bm * bm_M_cm * cm_M_c).rotation)
            # Rot = (c_M_b * b_M_bm * bm_M_cm * cm_M_c).rotation
            # r = R.from_matrix(Rot)
            # q = R.as_quat(r)
            # res[6*i+3:6*i+6] = q[0:3]

            # res[9*i+3:9*i+6] = c_M_b.rotation[:,0] - (b_M_bm * bm_M_cm * cm_M_c).inverse().rotation[:,0]
            # res[9*i+6:9*i+9] = c_M_b.rotation[:,1] - (b_M_bm * bm_M_cm * cm_M_c).inverse().rotation[:,1]
            # res[3*i:3*i+3] = c_M_b.translation - (cm_M_c.inverse() * bm_M_cm.inverse() * bm_M_b).translation
        self.cost_arr.append(np.linalg.norm(res))

        return res

    

if __name__ == '__main__':

    alias = 'legrand1'
    data_path = 'data/'

    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    df_gt = pd.read_pickle(data_path + f'groundtruth_{alias}.pkl')
    mocap_wrapper = MocapWrapper(df_gt)

    nu_c_list = []
    nu_b_list = []

    n_iter = 1
    n_samples = 300

    for i in range(n_iter):
        if n_iter == 1:
            df_cosypose_sample = df_cosypose
        else:
            df_cosypose_sample = df_cosypose.sample(n=n_samples, random_state=i)


        # cosypose trajectory
        counter = 0
        c_M_b_traj = [pin.SE3(T) for T in df_cosypose_sample['pose']]
        N = len(c_M_b_traj)
        bm_M_cm_traj,_ = mocap_wrapper.trajectory_generation(df_cosypose_sample)

        # priors
        cost = CostFrameCalibration(bm_M_cm_traj, c_M_b_traj)
        
        x0 = np.zeros(12)  # chosen to be [nu_c, nu_b]
        for i in range(2):
            r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)
            x0 = r.x

        nu_c = r.x[:6]
        nu_b = r.x[6:]
        nu_c_list.append(nu_c)
        nu_b_list.append(nu_b)

    # recover the transformation with exp
    cm_M_c_est = pin.exp6(nu_c) 
    bm_M_b_est = pin.exp6(nu_b)
    print('cm_M_c_est\n', cm_M_c_est)
    print('bm_M_b_est\n', bm_M_b_est)

    np.savez('nu_list.npz', nu_c=nu_c_list, nu_b = nu_b_list)
    np.savez(data_path+f'calibration_{alias}.npz', cm_M_c=cm_M_c_est, bm_M_b=bm_M_b_est)
    
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
