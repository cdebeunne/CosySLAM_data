import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

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