#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from utils.wrapper import MocapWrapper
import utils.posemath as pm

"""
Extrinsic calibration between mocap and cosypose by minimizing
the translation and rotation error on a set of trajectories

Returns cmMc and bmMb in a npz file
"""


class CostFrameCalibration:
    def __init__(self, m_M_cm_traj, m_M_bm_traj, c_M_b_cosy_traj):
        self.x_arr = []
        self.cost_arr = []
        self.m_M_cm_traj = m_M_cm_traj
        self.m_M_bm_traj = m_M_bm_traj
        self.c_M_b_cosy_traj = c_M_b_cosy_traj
        self.N = len(m_M_cm_traj)
        self.N_res = 6*self.N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        nu_c = x[:6]  # currently estimated transfo as se3 6D vector representation
        nu_b = x[6:]  # currently estimated transfo as se3 6D vector representation
        cm_M_c = pin.exp6(nu_c)  # currently estimated transfo as SE3 Lie group
        bm_M_b = pin.exp6(nu_b)  # currently estimated transfo as SE3 Lie group
        b_M_bm = bm_M_b.inverse()
        res = np.zeros(self.N_res)
        for i in range(self.N):
            m_M_cm = self.m_M_cm_traj[i]
            m_M_bm = self.m_M_bm_traj[i]
            bm_M_cm = m_M_bm.inverse() * m_M_cm
            c_M_b  = self.c_M_b_cosy_traj[i]
            b_M_b = b_M_bm * bm_M_cm * cm_M_c * c_M_b
            c_M_c = c_M_b * b_M_bm * bm_M_cm * cm_M_c
            m_M_m = m_M_cm * cm_M_c * c_M_b * b_M_bm * m_M_bm.inverse()
            res[6*i:6*i+3]   = c_M_c.translation*20
            res[6*i+3:6*i+6] = pin.log3(c_M_c.rotation)
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
    
    aliases = ['switchBIS3','switchBIS4','switchBIS5']
    aliases = ['legrand1','legrand2', 'legrand4']
    aliases = ['newLegrand1', 'newLegrand2', 'newLegrand3', 'newLegrand5']
    aliases = ['switch1','switch4','switch5']
    aliases = ['campbell1', 'campbell2', 'campbell3']
    data_path = 'data/'

    dfs_cosypose = []
    mocap_wrappers = []
    for alias in aliases:
        df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
        df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
        df_gt = pd.read_pickle(data_path + f'groundtruth_{alias}.pkl')
        mocap_wrapper = MocapWrapper(df_gt)
        dfs_cosypose.append(df_cosypose)
        mocap_wrappers.append(mocap_wrapper)

    nu_c_list = []
    nu_b_list = []

    n_iter = 1
    n_samples = 150

    for i in range(n_iter):
        
        c_M_b_traj = []
        bm_M_cm_traj = []
        m_M_cm_traj = []
        m_M_bm_traj = []

        for j in range (len(aliases)):
            mocap_wrapper = mocap_wrappers[j]
            df_cosypose = dfs_cosypose[j]

            if n_iter == 1:
                df_cosypose_sample = df_cosypose
            else:
                # df_cosypose_sample = pd.concat([df_cosypose[1:].sample(n=n_samples, random_state=i),df_cosypose[0:1]])
                df_cosypose_sample = df_cosypose

            # cosypose trajectory
            c_M_b_traj += [pin.SE3(T) for T in df_cosypose_sample['pose']]
            bm_M_cm_traj_j, m_M_cm_traj_j, m_M_bm_traj_j,_ = mocap_wrapper.trajectory_generation(df_cosypose_sample)
            bm_M_cm_traj += bm_M_cm_traj_j
            m_M_cm_traj += m_M_cm_traj_j
            m_M_bm_traj += m_M_bm_traj_j

        # cost function
        cost = CostFrameCalibration(m_M_cm_traj, m_M_bm_traj, c_M_b_traj)
        N = len(c_M_b_traj)
        x0 = np.zeros(12)  # chosen to be [nu_c, nu_b]

        # random init
        if n_iter > 1:
            cm_R_c = R.random().as_matrix()
            bm_R_b = R.random().as_matrix()
            cm_M_c = np.identity(4)
            cm_M_c[0:3,0:3] = cm_R_c
            bm_M_b = np.identity(4)
            bm_M_b[0:3,0:3] = bm_R_b
            x0 = np.array(pin.log6(pin.SE3(cm_M_c)).np.tolist()+pin.log6(pin.SE3(bm_M_b)).np.tolist())

        for j in range(2):
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
    print('Angular displacement :')
    print(pm.log3_to_euler(pin.log3(cm_M_c_est.rotation)))
    print('bm_M_b_est\n', bm_M_b_est)
    print('Angular displacement :')
    print(pm.log3_to_euler(pin.log3(bm_M_b_est.rotation)))

    np.savez('nu_list.npz', nu_c=nu_c_list, nu_b = nu_b_list)
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
