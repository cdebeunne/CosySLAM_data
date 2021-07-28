import numpy as np
import pinocchio as pin
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.wrapper import MocapWrapper

def compute_speed(M_traj, timestamp):
    """
    return the absolute speed of a given trajectory with its timestamp
    """
    first = True
    speed = []
    counter = 0
    for M in M_traj:
        if first:
            t = timestamp[counter]
            pt = M.translation
            first = False
            counter += 1
            continue
        tp1 = timestamp[counter]
        ptp1 = M.translation
        magn = np.linalg.norm((ptp1-pt)/(tp1-t))
        speed.append(magn)
        pt = ptp1
        t = tp1
        counter += 1
    return speed

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def mu_plot(nu_c_list, nu_b_list):
    """
    plot the dispersion of the calibration matrix
    """
    v1c = nu_c_list[:,0]
    v2c = nu_c_list[:,1]
    v3c = nu_c_list[:,2]
    w1c = nu_c_list[:,3]
    w2c = nu_c_list[:,4]
    w3c = nu_c_list[:,5]

    v1b = nu_b_list[:,0]
    v2b = nu_b_list[:,1]
    v3b = nu_b_list[:,2]
    w1b = nu_b_list[:,3]
    w2b = nu_b_list[:,4]
    w3b = nu_b_list[:,5]
    
    plt.figure()
    plt.plot('v1', [v1c], 'gx')
    plt.plot('v2', [v2c], 'gx')
    plt.plot('v3', [v3c], 'gx')
    plt.plot('w1', [w1c], 'rx')
    plt.plot('w2', [w2c], 'rx')
    plt.plot('w3', [w3c], 'rx')
    plt.title('Dispersion of R^6 coefficients of cm_M_c')

    plt.figure()
    plt.plot('v1', [v1b], 'gx')
    plt.plot('v2', [v2b], 'gx')
    plt.plot('v3', [v3b], 'gx')
    plt.plot('w1', [w1b], 'rx')
    plt.plot('w2', [w2b], 'rx')
    plt.plot('w3', [w3b], 'rx')
    plt.title('Dispersion of R^6 coefficients of bm_M_b')
    return

if __name__ == '__main__':
    
    alias = 'quick'
    data_path = 'data/'

    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    df_gt = pd.read_pickle(data_path + f'groundtruth_{alias}.pkl')
    mocap_wrapper = MocapWrapper(df_gt)

    # cosypose trajectory
    c_M_b_cosy = [pin.SE3(T[0]) for T in df_cosypose['pose']]
    
    # moCap trajectory, synchronized with cosypose's ts
    bm_M_cm_traj, timestamp = mocap_wrapper.trajectory_generation(df_cosypose)
    
    # loading calibration data
    calibration = np.load(data_path + f'calibration_{alias}.npz')
    cm_M_c = pin.SE3(calibration['cm_M_c'])
    bm_M_b = pin.SE3(calibration['bm_M_b'])

    # correcting the transformation wrt the calibration
    c_M_b_mocap = [cm_M_c.inverse() * bm_M_cm.inverse() * bm_M_b for bm_M_cm in bm_M_cm_traj]

    # trajectory to plot
    poseMocap = np.array([c_M_b.inverse().translation for c_M_b in c_M_b_mocap])
    poseCosy = np.array([c_M_b.inverse().translation for c_M_b in c_M_b_cosy])
    angleMocap = np.array([pin.log3(c_M_b.inverse().rotation) for c_M_b in c_M_b_mocap])
    angleCosy = np.array([pin.log3(c_M_b.inverse().rotation) for c_M_b in c_M_b_cosy])
    transNorm_mocap = [np.linalg.norm(t) for t in poseMocap]
    transNorm_cosy = [np.linalg.norm(t) for t in poseCosy]
    cosy_speed = compute_speed(c_M_b_cosy, timestamp)
    mocap_speed = compute_speed(c_M_b_mocap, timestamp)

    # error values
    trans_err = np.abs(poseMocap - poseCosy)
    rot_err = np.abs(angleMocap - angleCosy)
    rmse_trans = np.mean([np.linalg.norm(err) for err in trans_err])
    rmse_rot = np.mean([np.linalg.norm(err) for err in rot_err])
    print('rmse translation :' + str(rmse_trans))
    print('rmse rotation : ' + str(rmse_rot*180/3.14))

    # loading coefficient dispersion
    data = np.load('nu_list.npz')
    nu_c_list = data['nu_c']
    nu_b_list = data['nu_b']
  
    # Let's plot!
    # plt.figure('Velocity comparison')
    # plt.plot(timestamp[1:], cosy_speed, label='cosypose')
    # plt.plot(timestamp[1:], mocap_speed, label='mocap')
    # plt.legend()

    # fig = plt.figure('Trajectory comparison')
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(poseMocap[:,0], poseMocap[:,1], poseMocap[:,2], label='mocap')
    # ax.scatter(poseCosy[:,0], poseCosy[:,1], poseCosy[:,2], label='cosypose')
    # plt.legend()

    plt.figure('Translation comparison')
    plt.plot(timestamp, transNorm_cosy, label='cosypose')
    plt.plot(timestamp, transNorm_mocap, label='mocap')
    plt.legend()

    plt.figure('Translation error')
    ax1 = plt.subplot(131)
    ax1.plot(poseMocap[:,0], label='mocap')
    ax1.plot(poseCosy[:,0], label='cosy')
    ax2 = plt.subplot(132)
    ax2.plot(poseMocap[:,1], label='mocap')
    ax2.plot(poseCosy[:,1], label='cosy')
    ax3 = plt.subplot(133)
    ax3.plot(poseMocap[:,2], label='mocap')
    ax3.plot(poseCosy[:,2], label='cosy')
    plt.legend()

    plt.figure('Rotation error')
    ax1 = plt.subplot(131)
    ax1.plot(angleMocap[:,0], label='mocap')
    ax1.plot(angleCosy[:,0], label='cosy')
    ax2 = plt.subplot(132)
    ax2.plot(angleMocap[:,1], label='mocap')
    ax2.plot(angleCosy[:,1], label='cosy')
    ax3 = plt.subplot(133)
    ax3.plot(angleMocap[:,2], label='mocap')
    ax3.plot(angleCosy[:,2], label='cosy')
    plt.legend()

    # mu_plot(nu_c_list, nu_b_list)
    plt.show()
  