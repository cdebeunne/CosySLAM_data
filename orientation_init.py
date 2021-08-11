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
    a_g = np.transpose(np.array([[0,0,9.81]]))

    # a_meas is the average acceleration on N measurements
    a_meas = np.array([0.0,0.0,0.0])
    N = int(sys.argv[2])
    N_IMU = bag.get_message_count('/imu')
    counter = 0
    for topic, msg, t in bag.read_messages(topics=['/imu']):
        if (counter > N_IMU-N):
            a_t = np.array([msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z])
            a_meas += a_t
        counter += 1
    a_meas = np.transpose([(1/N)*a_meas])
    # a_meas = np.transpose(np.array([[0.0,9.81,0.0]]))
    print('a_meas')
    print(a_meas)
    print('a_meas norm')
    print(np.linalg.norm(a_meas))

    # solving with linear algebra
    M = a_meas @ np.transpose(a_g)
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

    # We then solve the same problem with the Rodriguz formula
    a_g = a_g/np.linalg.norm(a_g)
    a_meas = a_meas/np.linalg.norm(a_meas)
    v = np.cross(np.transpose(a_g)[0], np.transpose(a_meas)[0])
    c = np.dot(np.transpose(a_g)[0], np.transpose(a_meas)[0])
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



