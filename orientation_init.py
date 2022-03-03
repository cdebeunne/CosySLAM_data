from math import degrees

from numpy import linalg
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
import utils.posemath as pm
import rosbag
import rospy
import sys

'''
Gives the initial orientation of an IMU

arg 1 : alias of the rosbag
arg 2 : number of IMU sample
'''
SIMU = False

if __name__ == '__main__':

    # bag opening 
    alias = sys.argv[1]
    bag = rosbag.Bag(f'{alias}.bag', 'r')
    N = int(sys.argv[2])
    N_IMU = bag.get_message_count('/imu')
    assert(N <= N_IMU)

    # w_g is the gravitationnal vector
    w_g = np.array([0,0,-9.81])
    # w_g = np.array([1,1,1])
    w_g_mat = np.tile(w_g, N).reshape((N,3))


    if not SIMU:

        a_meas_arr = []
        # fill the measurement list
        for counter, (topic, msg, t) in enumerate(bag.read_messages(topics=['/imu'])):
            if counter < N:
                a_meas_arr.append([msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z])
        
        # shape : (number of samples, 3) 
        a_meas_arr = np.array(a_meas_arr)   

    if SIMU:
        # Simulated data instead
        angle = np.deg2rad(90)
        dir = np.random.random(3)
        dir = np.array([1,0,0])
        dir = dir/np.linalg.norm(dir)
        wRb_gtr = R.from_rotvec(angle*dir).as_matrix()
        print('\nwRb_gtr\n', wRb_gtr)
        r_gtr = R.from_matrix(wRb_gtr)
        print('As euler angles :')
        print(r_gtr.as_euler('xyz', degrees=True))
        print('As quaternion : ')
        print(r_gtr.as_quat())
        a_meas_arr = -wRb_gtr.T @ w_g_mat.T  # + noise  # # (3,N)
        a_meas_arr = a_meas_arr.T  # (N,3)

    

    # we solve an orthogonal Procrustes problem 
    # b_R_w = argmin R || R*a_g - a_meas ||
    # solving with linear algebra (https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
    # Wikipedia notations: 
    A = -w_g_mat.T
    B = a_meas_arr.T
    M = B @ A.T
    U, sig, V_T = np.linalg.svd(M)
    b_R_w = U @ V_T

    # display the result!
    print('\nProcrustes')
    print('b_R_w :')
    print(b_R_w)
    r_pro = R.from_matrix(b_R_w)
    print('As euler angles :')
    print(r_pro.as_euler('xyz', degrees=True))
    print('As quaternion : ')
    print(r_pro.as_quat())
    print('Is it aligned ?')
    avg_acc = np.mean(a_meas_arr, axis=0)
    angle = pm.angle_between_vecs(avg_acc, -b_R_w@w_g)
    print(np.rad2deg(angle))
    print('We need the quaternion of w_R_b :')
    print(r_pro.inv().as_quat())

    # We then solve the same problem with the Rodriguez formula
    a_mean = np.mean(a_meas_arr, axis=0)
    w_g_norm = w_g/np.linalg.norm(w_g)
    a_norm = a_mean/np.linalg.norm(a_mean)
    v = np.cross(-w_g_norm, a_norm)
    c = np.dot(-w_g_norm, a_norm)
    s = np.linalg.norm(v)
    v_skew = np.array([[0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]])

    b_R_w = np.identity(3) + v_skew + (v_skew @ v_skew)/(1+c)

    # display the result!
    print('\nRodriguez')
    print('b_R_w :')
    print(b_R_w)
    r_rodr = R.from_matrix(b_R_w)
    print('As euler angles :')
    print(r_rodr.as_euler('xyz', degrees=True))
    print('As quaternion : ')
    print(r_rodr.as_quat())
    print('Is it aligned ?')
    angle = pm.angle_between_vecs(avg_acc, -b_R_w@w_g)
    print(np.rad2deg(angle))
    print(b_R_w.T @ avg_acc)
    print('We need the quaternion of w_R_b :')
    print(r_rodr.inv().as_quat())



    print('\nProcrustes vs Rodriguez \o--o/ ')
    rdiff_rodr_pro = r_rodr*r_pro.inv()
    print(rdiff_rodr_pro.as_rotvec())
    print(np.rad2deg(np.linalg.norm(rdiff_rodr_pro.as_rotvec())))

    if SIMU:
        rdiff_pro_gtr = r_pro*r_gtr.inv()
        print('\nProcrustes vs Ground truth \o--o/ ')
        print(rdiff_pro_gtr.as_rotvec())
        print(np.rad2deg(np.linalg.norm(rdiff_pro_gtr.as_rotvec())))

        rdiff_rodr_gtr = r_rodr*r_gtr.inv()
        print('\Rodriguez vs Ground truth \o--o/ ')
        print(rdiff_rodr_gtr.as_rotvec())
        print(np.rad2deg(np.linalg.norm(rdiff_rodr_gtr.as_rotvec())))

