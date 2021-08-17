import numpy as np
import pinocchio as pin
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import rospy

def log3_to_euler(omega):
    rot_mat = pin.exp3(omega)
    r = R.from_matrix(rot_mat)
    angles = r.as_euler('zyx', degrees=True)
    return angles

def isometry_to_vec(M):
    r = R.from_matrix(M.rotation)
    q = r.as_quat()
    vec = np.concatenate((M.translation, q))
    return vec

def isometry_to_transform(M):

    # Deal with translation
    translation_vector = M.translation
    translation = Point()
    translation.x = translation_vector[0]
    translation.y = translation_vector[1]
    translation.z = translation_vector[2]

    # Deal with rotation
    rotation = R.from_matrix(M.rotation)
    quat_vector = R.as_quat(rotation)
    quat = Quaternion()
    quat.x = quat_vector[0]
    quat.y = quat_vector[1]
    quat.z = quat_vector[2]
    quat.w = quat_vector[3]

    transf = Pose(translation, quat)
    return transf

def angle_between_vecs(a, b):
    cross = np.cross(a, b)
    sintheta = np.linalg.norm(cross)/(np.linalg.norm(a)*np.linalg.norm(b))
    angle = np.arcsin(sintheta)
    return angle