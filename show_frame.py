import numpy as np
import pinocchio as pin
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mocap_objects_frame_calibration import mocapTraj_generator
import gepetto.corbaserver
from gepetto.corbaserver import tools
import time

def GepettoViewerServer(windowName="python-pinocchio", sceneName="world", loadModel=False):
    """
    Init gepetto-viewer by loading the gui and creating a window.
    """
    viewer = gepetto.corbaserver.Client()
    gui = viewer.gui

    # Create window
    window_l = gui.getWindowList()
    if windowName not in window_l:
        gui.windowID = gui.createWindow(windowName)
    else:
        gui.windowID = gui.getWindowID(windowName)

    # Create scene if needed
    scene_l = gui.getSceneList()
    if sceneName not in scene_l:
        gui.createScene(sceneName)
        gui.addSceneToWindow(sceneName, gui.windowID)

    gui.sceneName = sceneName

    return gui

def isometry_to_vec(M):
    r = R.from_matrix(M.rotation)
    q = r.as_quat()
    vec = np.concatenate((M.translation, q))
    return vec


if __name__ == '__main__':

    alias = 'calib_donut'
    data_path = 'data/'

    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    df_gt = pd.read_pickle(data_path + f'groundtruth_{alias}.pkl')

    # loading calibration data
    calibration = np.load(data_path+'calibration_donut.npz')
    cm_M_c = pin.SE3(calibration['cm_M_c'])
    bm_M_b = pin.SE3(calibration['bm_M_b'])

    # selecting the frames of interest
    df_cosypose = df_cosypose
    bm_M_cm_traj,_ = mocapTraj_generator(df_gt, df_cosypose)

    # correcting the transformation wrt the calibration
    c_M_b_mocap = [cm_M_c.inverse() * bm_M_cm.inverse() * bm_M_b for bm_M_cm in bm_M_cm_traj]
    c_M_b_cosy = [pin.SE3(T[0]) for T in df_cosypose['pose']]

    # connect the gui server
    gui = GepettoViewerServer(windowName="tuto")
    gui.createGroup ("mocap")
    gui.addToGroup ("mocap", "tuto")

    gui.createGroup ("cosy")
    gui.addToGroup ("cosy", "tuto")
    time.sleep(1)

    for i in range(1, 150):
        print("frame nbr :" + str(i))
        vec_mocap = isometry_to_vec(c_M_b_mocap[i])
        gui.addLandmark ("mocap", 0.1)
        gui.applyConfiguration("mocap", vec_mocap.tolist())
        frame_mocap = tools.Vector6("mocap")
        frame_mocap.create(gui)

        vec_cosy = isometry_to_vec(c_M_b_cosy[i])
        gui.addLandmark ("cosy", 0.5)
        gui.applyConfiguration("cosy", vec_cosy.tolist())
        frame_mocap = tools.Vector6("cosy")
        frame_mocap.create(gui)

        time.sleep(0.05)
        gui.refresh()
    
    while True:
        time.sleep(10)
        gui.refresh()
