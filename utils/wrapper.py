import numpy as np
import pinocchio as pin

class MocapWrapper:
    def __init__(self, df_mocap):

        self.df_mocap = df_mocap
        # timestamp setting
        self.df_mocap['smTimestamp'] -= self.df_mocap['smTimestamp'][0]
        self.df_mocap['cmTimestamp'] -= self.df_mocap['cmTimestamp'][0]

    def trajectory_generation(self, df_cosypose):
        """
        returns the mocap trajectory synchronized with the camera timestamp
        optionnaly add a delta_t if the ts is not synchro
        """
        # timestamp setting
        df_cosypose['timestamp'] -= df_cosypose['timestamp'][0]

        # variable init
        timestamp = []
        bm_M_cm_traj = []

        # moCap trajectory, synchronized with cosypose's ts
        for ts in df_cosypose['timestamp']:
            timestamp.append(float(ts))
            idxSm = self.df_mocap['smTimestamp'].sub(float(ts)).abs().idxmin()
            m_M_bm = pin.SE3(self.df_mocap.loc[idxSm,'mTsm'])
            idxCm = self.df_mocap['cmTimestamp'].sub(float(ts)).abs().idxmin()
            m_M_cm = pin.SE3(self.df_mocap.loc[idxCm,'mTcm'])
            bm_M_cm_traj.append(m_M_bm.inverse() * m_M_cm)

        return bm_M_cm_traj, timestamp