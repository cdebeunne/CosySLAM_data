import numpy as np
import pinocchio as pin
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.wrapper import MocapWrapper
import gepetto.corbaserver
from gepetto.corbaserver import tools
import time
import utils.posemath as pm

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


if __name__ == '__main__':

    alias = 'campbell2'
    data_path = 'data/'

    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    df_gt = pd.read_pickle(data_path + f'groundtruth_{alias}.pkl')
    mocap_wrapper = MocapWrapper(df_gt)

    # loading calibration data
    calibration = np.load(data_path+'calibration_campbell.npz')
    cm_M_c = pin.SE3(calibration['cm_M_c'])
    bm_M_b = pin.SE3(calibration['bm_M_b'])

    # selecting the frames of interest
    df_cosypose = df_cosypose
    bm_M_cm_traj, m_M_cm_traj, m_M_bm_traj,_ = mocap_wrapper.trajectory_generation(df_cosypose)

    # correcting the transformation wrt the calibration
    c_M_b_mocap = [cm_M_c.inverse() * bm_M_cm.inverse() * bm_M_b for bm_M_cm in bm_M_cm_traj]
    c_M_b_cosy = [pin.SE3(T) for T in df_cosypose['pose']]

    # connect the gui server
    gui = GepettoViewerServer(windowName="tuto")
    gui.createGroup ("bm")
    gui.addToGroup ("bm", "tuto")

    gui.createGroup ("cm")
    gui.addToGroup ("cm", "tuto")

    gui.createGroup ("b_cosy")
    gui.addToGroup ("b_cosy", "tuto")

    gui.createGroup ("c_cosy")
    gui.addToGroup ("c_cosy", "tuto")
    time.sleep(1)

    for i in range(1, 450):
        print("frame nbr :" + str(i))
        vec_bm = pm.isometry_to_vec(m_M_bm_traj[i])
        gui.addLandmark ("bm", 0.1)
        gui.applyConfiguration("bm", vec_bm.tolist())
        frame_mocap = tools.Vector6("bm")
        frame_mocap.create(gui)

        vec_cm = pm.isometry_to_vec(m_M_cm_traj[i])
        gui.addLandmark ("cm", 0.1)
        gui.applyConfiguration("cm", vec_cm.tolist())
        frame_mocap = tools.Vector6("cm")
        frame_mocap.create(gui)

        m_M_b_cosy = m_M_bm_traj[i] * bm_M_b
        vec_b_cosy = pm.isometry_to_vec(m_M_b_cosy)
        gui.addLandmark ("b_cosy", 0.1)
        gui.applyConfiguration("b_cosy", vec_b_cosy.tolist())
        frame_mocap = tools.Vector6("b_cosy")
        frame_mocap.create(gui)

        m_M_c_cosy = m_M_cm_traj[i] * cm_M_c
        vec_c_cosy = pm.isometry_to_vec(m_M_c_cosy)
        gui.addLandmark ("c_cosy", 0.1)
        gui.applyConfiguration("c_cosy", vec_c_cosy.tolist())
        frame_mocap = tools.Vector6("c_cosy")
        frame_mocap.create(gui)

        time.sleep(0.05)
        gui.refresh()
    
    while True:
        time.sleep(10)
        gui.refresh()
