from math import degrees
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
import rosbag
import rospy
import sys

if __name__ == '__main__':

    # bag opening 
    alias = sys.argv[1]
    bag = rosbag.Bag(f'{alias}.bag', 'r')

    # we solve an orthogonal Procrustes problem 
    # b_R_w = argmin R || R*a_g - (-a_meas) ||
    
    # a_g is the gravitationnal vector
    a_g = np.transpose(np.array([[0,0,-9.81]]))

    # a_meas is the average acceleration on N measurements
    a_meas = np.array([0.0,0.0,0.0])
    N = int(sys.argv[2])
    counter = 0
    for topic, msg, t in bag.read_messages(topics=['/imu']):
        if (counter > N-1):
            break
        a_t = np.array([msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z])
        a_meas += a_t
        counter += 1
    a_meas = np.transpose([(1/N)*a_meas])
    print('a_meas normalized')
    print(a_meas/np.linalg.norm(a_meas))

    # solving with linear algebra
    M = - a_meas @ np.transpose(a_g)
    U, sig, V_T = np.linalg.svd(M)
    b_R_w = U @ V_T

    # display the result!
    print('b_R_w :')
    print(b_R_w)
    r = R.from_matrix(b_R_w)
    print('As euler angles :')
    print(r.as_euler('zyx', degrees=True))
    print('As quaternion : ')
    print(r.as_quat())

