import numpy as np
import pinocchio as pin
import pandas as pd

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

class ErrorWrapper:
    def __init__(self, object_name, data_path):
        
        self.data_path = data_path
        self.object_name = object_name
        self.frame_id = []
        self.rotation = []
        self.detection_score = []
        self.translation_err_0 = []
        self.translation_err_1 = []
        self.translation_err_2 = []
        self.angular_err_0 = []
        self.angular_err_1 = []
        self.angular_err_2 = []
        self.r_list = []
        self.phi_list = []
        self.theta_list = []
        self.scene = []
    
    def create_df(self, aliases):
        delta_count = 0
        for alias in aliases:
            calibration_path = self.data_path + f'calibration_{alias}.npz'

            df_cosypose = pd.read_pickle(self.data_path+'results_{}_ts.pkl'.format(alias))
            df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
            df_cosypose = df_cosypose.loc[df_cosypose['object_name'] == self.object_name]
            df_cosypose = df_cosypose.loc[df_cosypose['detection_score'] > 0.98]
            df_gt = pd.read_pickle(self.data_path + 'groundtruth_{}.pkl'.format(alias))
            mocap_wrapper = MocapWrapper(df_gt)

            # cosypose trajectory
            c_M_b_cosy = [pin.SE3(T) for T in df_cosypose['pose']]
            cosy_score = df_cosypose['detection_score'].values
    
            # moCap trajectory, synchronized with cosypose's ts
            bm_M_cm_traj, _ = mocap_wrapper.trajectory_generation(df_cosypose)
            delta_count += 1
            
            # loading calibration data
            calibration = np.load(calibration_path)
            cm_M_c = pin.SE3(calibration['cm_M_c'])
            bm_M_b = pin.SE3(calibration['bm_M_b'])

            # correcting the transformation wrt the calibration
            c_M_b_mocap = [cm_M_c.inverse() * bm_M_cm.inverse() * bm_M_b for bm_M_cm in bm_M_cm_traj]

            counter = 0
            for c_M_b in c_M_b_cosy:
                b_M_c = c_M_b.inverse()
                b_M_c_gt = c_M_b_mocap[counter].inverse()

                # X data
                self.rotation.append(b_M_c_gt.rotation)
                r = np.linalg.norm(b_M_c_gt.translation)
                self.r_list.append(r)
                phi = np.arcsin(b_M_c_gt.translation[2]/r)
                self.phi_list.append(phi)
                theta = np.arctan(b_M_c_gt.translation[0]/b_M_c_gt.translation[1])
                self.theta_list.append(theta)
                self.detection_score.append(cosy_score[counter])

                # error data
                translation_err = b_M_c_gt.rotation @ (b_M_c_gt.translation-b_M_c.translation)
                self.translation_err_0.append(translation_err[0])
                self.translation_err_1.append(translation_err[1])
                self.translation_err_2.append(translation_err[2])
                angular_err = pin.log3((b_M_c*b_M_c_gt.inverse()).rotation)
                self.angular_err_0.append(angular_err[0])
                self.angular_err_1.append(angular_err[1])
                self.angular_err_2.append(angular_err[2])

                self.frame_id.append(counter)
                self.scene.append(alias)
                counter += 1
        
        df = pd.DataFrame({
            'scene':self.scene,
            'frame_id':self.frame_id,
            'rotation':self.rotation,
            'detection_score':self.detection_score,
            'r':self.r_list,
            'phi':self.phi_list,
            'theta':self.theta_list,
            'translation_err_0':self.translation_err_0,
            'translation_err_1':self.translation_err_1,
            'translation_err_2':self.translation_err_2,
            'angular_err_0':self.angular_err_0,
            'angular_err_1':self.angular_err_1,
            'angular_err_2':self.angular_err_2,
        })
        return df