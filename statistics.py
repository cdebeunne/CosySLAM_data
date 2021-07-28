import numpy as np
from numpy.lib.polynomial import poly
from scipy.spatial.transform import Rotation as R

import pinocchio as pin
import pandas as pd
from utils.wrapper import MocapWrapper

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import joblib
import utils.posemath as pm

def bb_to_center(bb, resolution):
    x = (bb[0]+bb[2])/2
    y = (bb[1]+bb[3])/2
    x = np.abs(x-resolution[0]/2)/(resolution[0]/2)
    y = np.abs(y-resolution[1]/2)/(resolution[1]/2)
    return x,y


def create_df(obj_name, aliases):
    # let's initialize the lists and the scenes' name
    frame_id = []
    rotation = []
    speed_magnitude = []
    angular_speed_magnitude = []
    detection_score = []
    translation_err_0 = []
    translation_err_1 = []
    translation_err_2 = []
    angular_err_0 = []
    angular_err_1 = []
    angular_err_2 = []
    r_list = []
    phi_list = []
    theta_list = []
    scene = []

    delta_count = 0
    for alias in aliases:
        data_path = '{}/'.format(alias)
        calibration_path = data_path + 'calibration.npz'

        df_cosypose = pd.read_pickle(data_path+'results_{}_ts.pkl'.format(alias))
        df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
        df_cosypose = df_cosypose.loc[df_cosypose['object_name'] == obj_name]
        df_cosypose = df_cosypose.loc[df_cosypose['detection_score'] > 0.98]
        df_gt = pd.read_pickle(data_path + 'groundtruth_{}.pkl'.format(alias))
        mocap_wrapper = MocapWrapper(df_gt)

        # cosypose trajectory
        c_M_b_cosy = [pin.SE3(T) for T in df_cosypose['pose']]
        cosy_score = df_cosypose['detection_score'].values
        cosy_bb = df_cosypose['bbox'].values
        
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
        dt = 1/30
        for c_M_b in c_M_b_cosy:
            b_M_c = c_M_b.inverse()
            b_M_c_gt = c_M_b_mocap[counter].inverse()

            # X data
            rotation.append(b_M_c_gt.rotation)
            r = np.linalg.norm(b_M_c_gt.translation)
            r_list.append(r)
            phi = np.arcsin(b_M_c_gt.translation[2]/r)
            phi_list.append(phi)
            theta = np.arctan(b_M_c_gt.translation[0]/b_M_c_gt.translation[1])
            theta_list.append(theta)
            detection_score.append(cosy_score[counter])

            if counter == 0:
                speed_magnitude.append(0)
                angular_speed_magnitude.append(0)
            else:
                b_M_c_gt_prev = c_M_b_mocap[counter-1].inverse()
                speed = (b_M_c_gt_prev.translation-b_M_c_gt.inverse().translation)/dt
                speed_magnitude.append(np.linalg.norm(speed))
                angular_speed = pin.log3(b_M_c_gt_prev.rotation*b_M_c_gt.rotation)/dt
                angular_speed_magnitude.append(np.linalg.norm(angular_speed))

            # error data
            translation_err = b_M_c_gt.rotation @ (b_M_c_gt.translation-b_M_c.translation)
            translation_err_0.append(np.abs(translation_err[0]))
            translation_err_1.append(np.abs(translation_err[1]))
            translation_err_2.append(np.abs(translation_err[2]))
            angular_err = np.abs(pin.log3((b_M_c*b_M_c_gt.inverse()).rotation))
            angular_err_0.append(angular_err[0])
            angular_err_1.append(angular_err[1])
            angular_err_2.append(angular_err[2])

            frame_id.append(counter)
            scene.append(alias)
            counter += 1
    
    df = pd.DataFrame({
        'scene':scene,
        'frame_id':frame_id,
        'rotation':rotation,
        'detection_score':detection_score,
        'r':r_list,
        'phi':phi_list,
        'theta':theta_list,
        'speed':speed_magnitude,
        'angular_speed':angular_speed_magnitude,
        'u_list': u_list,
        'v_list': v_list,
        'translation_err_0':translation_err_0,
        'translation_err_1':translation_err_1,
        'translation_err_2':translation_err_2,
        'angular_err_0':angular_err_0,
        'angular_err_1':angular_err_1,
        'angular_err_2':angular_err_2,
    })
    return df

if __name__ == '__main__':

    aliases = ['legrand1', 'legrand2', 'legrand3', 'legrand4']
    aliases = ['switch1', 'switch2','switch4', 'switch5']
    df = create_df('obj_000026', aliases)

    # formatting the data 
    X = df[['r', 'theta', 'phi', 'detection_score']]
    Y = df[['translation_err_0', 'translation_err_1', 'translation_err_2',
        'angular_err_0', 'angular_err_1', 'angular_err_2']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    # let's set and train a polynomial regression pipeline 
    degree = 2
    model = LinearRegression()
    poly = PolynomialFeatures(degree)
    polyreg=make_pipeline(poly,model)
    polyreg.fit(X,Y)

    r = 0.28
    theta = 0.0
    phi = 1.0
    s = 0.99998
    pred = polyreg.predict([[r,theta,phi,s]])
    print(pred)
    print(pm.log3_to_euler(pred[0,3:]))

    # for degree 2 and 4 variables: 
    # [1 x1 x2 x3 x4 x1^2 x1*x2 x1*x3 x1*x4 x2^2 x2*x3 x2*x4 x3^2 x4*x3 x4^2]

    poly_coef = model.coef_[0]
    err0 = (poly_coef[0] + poly_coef[1]*r + poly_coef[2]*theta + poly_coef[3]*phi + poly_coef[4]*s +
            poly_coef[5]*r*r + poly_coef[6]*theta*r + poly_coef[7]*phi*r + poly_coef[8]*r*s +
            poly_coef[9]*theta*theta + poly_coef[10]*theta*phi + poly_coef[11]*theta*s +
            poly_coef[12]*phi*phi + poly_coef[13]*phi*s +
            poly_coef[14]*s*s +
            model.intercept_[0])
    print(err0)

    print("model score: "+str(polyreg.score(X, Y)))
    print("Model coeficients")
    np.set_printoptions(suppress=True)
    print(np.array2string(model.coef_, separator=', '))
    print(model.intercept_)

    # saving the model
    joblib.dump(polyreg, 'error_model.pkl', protocol=2)
    translation_err = np.sqrt(Y['translation_err_0']*Y['translation_err_0']+
                              Y['translation_err_1']*Y['translation_err_1']+
                              Y['translation_err_2']*Y['translation_err_2'])
    plt.plot(X['r'], translation_err, '*')
    plt.show()

