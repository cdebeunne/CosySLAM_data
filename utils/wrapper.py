import numpy as np
from pandas._libs.tslibs import timestamps
from scipy import optimize
import utils.posemath as pm
import pinocchio as pin
import pandas as pd
import rosbag
import rospy
import apriltag

class SLAMWrapper:
    def __init__(self, bag_slam, bag_mocap):
        self.bag_slam = bag_slam
        self.bag_mocap = bag_mocap
    
    def trajectory_generation(self):
        mocap_M_camera_traj = []
        ts_mocap = []
        slam_M_camera_traj = []
        ts_slam = []
        counter = 0
        t_lim = 1000

        for _, msg, t in self.bag_mocap.read_messages(topics=['/qualisys/realsense']):
            if (counter == 0):
                ts_init = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
                counter += 1
            translation = np.array([msg.position.x, msg.position.y, msg.position.z])
            quaternion = np.array([msg.orientation.x, msg.orientation.y,
                                msg.orientation.z, msg.orientation.w])
            trans_vec = np.concatenate((translation, quaternion))
            mocap_M_camera = pm.vec_to_isometry(trans_vec)
            mocap_M_camera_traj.append(mocap_M_camera)
            
            ts = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
            ts_mocap.append(ts-ts_init)
            if (ts-ts_init > t_lim): break

        for _, msg, t in self.bag_slam.read_messages(topics=['/robot_pose']):
            transform = msg.pose
            translation = np.array([transform.position.x, transform.position.y,
                                transform.position.z])
            quaternion = np.array([transform.orientation.x, transform.orientation.y,
                                transform.orientation.z, transform.orientation.w])
            trans_vec = np.concatenate((translation, quaternion))
            slam_M_camera = pm.vec_to_isometry(trans_vec)
            slam_M_camera_traj.append(slam_M_camera)

            ts = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
            ts_slam.append(ts)

        df_mocap = pd.DataFrame({
                "timestamp" : ts_mocap,
            })
        df_slam = pd.DataFrame({
                "timestamp" : ts_slam,
            })
        
        df_slam['timestamp'] = df_slam['timestamp'] - df_slam['timestamp'][0]
        df_mocap['timestamp'] = df_mocap['timestamp'] - df_mocap['timestamp'][0]
        
        # moCap trajectory, synchronized with slam's ts
        mocap_M_camera_traj_1 = []
        for ts in df_slam['timestamp']:
            idx = df_mocap['timestamp'].sub(float(ts)).abs().idxmin()
            mocap_M_camera = mocap_M_camera_traj[idx]
            mocap_M_camera_traj_1.append(mocap_M_camera)
        
        return mocap_M_camera_traj, mocap_M_camera_traj_1, slam_M_camera_traj, ts_slam

class ApriltagWrapper:
    def __init__(self, bag, tag_size, mtx, dist):
        self.bag = bag
        self.tag_size = tag_size
        self.mtx = mtx
        self.dist = dist
    
    def trajectory_generation(self):
        import cv2
        from cv_bridge import CvBridge
        bridge = CvBridge()
        timestamp = []
        pose_list = []

        for _, msg, t in self.bag.read_messages(topics=['/camera/color/image_raw']):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # BGR to GRAY
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # AprilTag detection
            options = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector(options)
            results = detector.detect(gray)
            num_detections = len(results)

            # Pose estimation
            imgPointsArr = []
            objPointsArr = []
            opointsArr = []

            if num_detections > 0:
                for i, detection in enumerate(results):
                    imagePoints = detection.corners.reshape(1,4,2)  

                    ob_pt1 = [-self.tag_size/2, -self.tag_size/2, 0.0]
                    ob_pt2 = [ self.tag_size/2, -self.tag_size/2, 0.0]
                    ob_pt3 = [ self.tag_size/2,  self.tag_size/2, 0.0]
                    ob_pt4 = [-self.tag_size/2,  self.tag_size/2, 0.0]
                    ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
                    object_pts = np.array(ob_pts).reshape(4,3)

                    opoints = np.array([
                        -1, -1, 0,
                        1, -1, 0,
                        1,  1, 0,
                        -1,  1, 0,
                        -1, -1, -2*1,
                        1, -1, -2*1,
                        1,  1, -2*1,
                        -1,  1, -2*1,
                    ]).reshape(-1, 1, 3) * 0.5*self.tag_size
                        
                    imgPointsArr.append(imagePoints)
                    objPointsArr.append(object_pts)
                    opointsArr.append(opoints)

                    # mtx - the camera calibration's intrinsics
                    _, rvec, tvec = cv2.solvePnP(object_pts, imagePoints, self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
                    rvec = np.array([rvec[0,0], rvec[1,0], rvec[2,0]])
                    tvec = np.array([tvec[0,0], tvec[1,0], tvec[2,0]])
                    rotMat,_ = cv2.Rodrigues(rvec)
                    pose = np.zeros((4,4))
                    pose[0:3,0:3] = rotMat
                    pose[0:3,3] = tvec
                    pose_list.append(pose)
                    timestamp.append(t.secs + t.nsecs*10e-10)
        df_apriltag = pd.DataFrame({
            "timestamp" : timestamp,
            "pose" : pose_list,
        })
        return df_apriltag

class MocapWrapper:
    def __init__(self, df_mocap):

        self.df_mocap = df_mocap
        # timestamp setting
        self.df_mocap['smTimestamp'] -= self.df_mocap['smTimestamp'][0]
        self.df_mocap['cmTimestamp'] -= self.df_mocap['cmTimestamp'][0]

    def trajectory_generation(self, df_cosypose):
        """
        returns the mocap trajectory synchronized with the camera timestamp
        """
        # timestamp setting
        df_cosypose['timestamp'] -= df_cosypose['timestamp'][0]

        # variable init
        timestamp = []
        bm_M_cm_traj = []
        m_M_cm_traj = []
        m_M_bm_traj = []

        # moCap trajectory, synchronized with cosypose's ts
        for ts in df_cosypose['timestamp']:
            timestamp.append(float(ts))
            idxSm = self.df_mocap['smTimestamp'].sub(float(ts)).abs().idxmin()
            m_M_bm = pin.SE3(self.df_mocap.loc[idxSm,'mTsm'])
            idxCm = self.df_mocap['cmTimestamp'].sub(float(ts)).abs().idxmin()
            m_M_cm = pin.SE3(self.df_mocap.loc[idxCm,'mTcm'])
            bm_M_cm_traj.append(m_M_bm.inverse() * m_M_cm)
            m_M_cm_traj.append(m_M_cm)
            m_M_bm_traj.append(m_M_bm)

        return bm_M_cm_traj, m_M_cm_traj, m_M_bm_traj, timestamp

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
        """
        creates the dataframe for error statistics
        """
        delta_count = 0
        for alias in aliases:
            calibration_path = self.data_path + f'calibration_{alias[:-1]}.npz'

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
            bm_M_cm_traj, _,_,_= mocap_wrapper.trajectory_generation(df_cosypose)
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