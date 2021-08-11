from math import degrees
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
import rosbag
import rospy
import sys

'''
arg 1 : alias of the rosbag
arg 2 : number of IMU sample
'''

if __name__ == '__main__':

    # bag opening 
    alias = sys.argv[1]
    bag = rosbag.Bag(f'{alias}.bag', 'r')

    # we solve an orthogonal Procrustes problem 
    # b_R_w = argmin R || R*a_g - a_meas ||
    
    # a_g is the gravitationnal vector
    w_g = np.array([0,0,-9.81])

    # a_meas is the average acceleration on N measurements
    a_meas_arr = []
    N = int(sys.argv[2])
    N_IMU = bag.get_message_count('/imu')
    assert(N <= N_IMU)
    # fill the measurement list
    for counter, (topic, msg, t) in enumerate(bag.read_messages(topics=['/imu'])):
        if counter < N:
            a_meas_arr.append([msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z])
    
    # shape : (number of samples, 3) 
    a_meas_arr = np.array(a_meas_arr)   

    # solving with linear algebra (https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
    # Wikipedia notations: 
    A = np.tile(-w_g, N).reshape((N,3)).T
    B = a_meas_arr.T
    M = B @ A.T
    U, sig, V_T = np.linalg.svd(M)
    b_R_w = U @ V_T

    # display the result!
    print('b_R_w :')
    print(b_R_w)
    r = R.from_matrix(b_R_w)
    print('As euler angles :')
    print(r.as_euler('xyz', degrees=True))
    print('As quaternion : ')
    print(r.as_quat())

    # We then solve the same problem with the Rodriguez formula
    a_mean = np.mean(a_meas_arr, axis=0)
    w_g_norm = w_g/np.linalg.norm(w_g)
    a_norm = a_mean/np.linalg.norm(a_mean)
    v = np.cross(w_g_norm, a_norm)
    c = np.dot(w_g_norm, a_norm)
    s = np.linalg.norm(v)
    v_skew = np.array([[0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]])

    b_R_w = np.identity(3) + v_skew + (v_skew @ v_skew)/(1+c)

    # display the result!
    print('b_R_w :')
    print(b_R_w)
    r = R.from_matrix(b_R_w)
    print('As euler angles :')
    print(r.as_euler('xyz', degrees=True))
    print('As quaternion : ')
    print(r.as_quat())



